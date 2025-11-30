import os
import math
import random
import time
import argparse
import yaml
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from torchcrf import CRF
except Exception:
    CRF = None

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Toy dataset (CoNLL-like)
# -----------------------
class ToyNERDataset(Dataset):
    def __init__(self, sentences: List[List[str]], tags: List[List[str]], vocab, tag2id, max_len=64):
        self.sentences = sentences
        self.tags = tags
        self.vocab = vocab
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx]
        labels = self.tags[idx]
        ids = [self.vocab.get(w.lower(), self.vocab["<unk>"]) for w in words][: self.max_len]
        tag_ids = [self.tag2id[t] for t in labels][: self.max_len]
        mask_len = len(ids)
        # pad
        ids += [self.vocab["<pad>"]]*(self.max_len - mask_len)
        tag_ids += [self.tag2id["O"]]*(self.max_len - mask_len)
        mask = [1]*mask_len + [0]*(self.max_len - mask_len)
        return torch.tensor(ids), torch.tensor(tag_ids), torch.tensor(mask, dtype=torch.uint8)


def build_toy_data() -> Tuple[List[List[str]], List[List[str]]]:
    sents = [
        ["John", "lives", "in", "New", "York", "."],
        ["Apple", "released", "iPhone", "in", "California", "."],
        ["Mary", "visited", "Paris", "last", "year", "."],
    ]
    tags = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-ORG", "O", "O", "O", "B-LOC", "O"],
        ["B-PER", "O", "B-LOC", "O", "O", "O"],
    ]
    return sents, tags


# -----------------------
# BiLSTM-CRF Model
# -----------------------
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=100, hidden_dim=256, dropout=0.5, use_crf=True):
        super().__init__()
        self.use_crf = use_crf and (CRF is not None)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, mask):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        emissions = self.fc(out)
        return emissions

    def loss(self, emissions, tags, mask):
        if self.use_crf:
            # maximize log-likelihood => minimize negative
            return -self.crf(emissions, tags, mask=mask.bool(), reduction='mean')
        else:
            # fall back to cross entropy with mask
            B, T, C = emissions.shape
            emissions = emissions.reshape(B*T, C)
            tags = tags.reshape(B*T)
            mask = mask.reshape(B*T).bool()
            return nn.CrossEntropyLoss()(emissions[mask], tags[mask])

    def decode(self, emissions, mask):
        if self.use_crf:
            return self.crf.decode(emissions, mask=mask.bool())
        else:
            # greedy decode
            return emissions.argmax(-1).tolist()


# -----------------------
# Training & Eval
# -----------------------

def train_one_epoch(model, loader, optim):
    model.train()
    total = 0.0
    for ids, tags, mask in loader:
        ids, tags, mask = ids.to(dev), tags.to(dev), mask.to(dev)
        emissions = model(ids, mask)
        loss = model.loss(emissions, tags, mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item()
    return total / max(1, len(loader))


def eval_f1(model, loader, id2tag):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for ids, tags, mask in loader:
            ids, tags, mask = ids.to(dev), tags.to(dev), mask.to(dev)
            emissions = model(ids, mask)
            pred_paths = model.decode(emissions, mask)
            # compute token-level F1 (simplified)
            for b in range(tags.size(0)):
                gold = tags[b][: mask[b].sum()].cpu().tolist()
                pred = pred_paths[b][: len(gold)]
                for g, p in zip(gold, pred):
                    if id2tag[g] != 'O' and id2tag[p] != 'O':
                        if g == p:
                            tp += 1
                        else:
                            fp += 1; fn += 1
                    elif id2tag[g] != 'O' and id2tag[p] == 'O':
                        fn += 1
                    elif id2tag[g] == 'O' and id2tag[p] != 'O':
                        fp += 1
    precision = tp / (tp + fp + 1e-8)
    recall = fn and tp / (tp + fn + 1e-8) or (tp and 1.0 or 0.0)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8) if (precision+recall) > 0 else 0.0
    return precision, recall, f1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config/bilstm_crf.yaml', help='Path to YAML config')
    ap.add_argument('--use_toy', action='store_true', help='Force use toy data regardless of config paths')
    return ap.parse_args()


def load_config(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Config not found: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def maybe_load_conll(cfg):
    from utils_data import try_read, build_vocab, build_tagset
    data_cfg = cfg.get('data', {})
    train_path = data_cfg.get('train')
    if train_path and os.path.isfile(train_path):
        res = try_read(train_path)
        if res:
            sents, tags = res
            vocab = build_vocab(sents)
            tag2id, id2tag = build_tagset(tags)
            return sents, tags, vocab, tag2id, id2tag
    return None


def prepare_data(cfg, use_toy=False):
    if not use_toy:
        loaded = maybe_load_conll(cfg)
        if loaded:
            return loaded
    # fallback toy
    sents, tags = build_toy_data()
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in sents:
        for w in s:
            lw = w.lower()
            if lw not in vocab:
                vocab[lw] = len(vocab)
    tagset = sorted({t for seq in tags for t in seq} | {"O"})
    tag2id = {t: i for i, t in enumerate(tagset)}
    id2tag = {i: t for t, i in tag2id.items()}
    return sents, tags, vocab, tag2id, id2tag


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.get('seed', 42))
    torch.manual_seed(cfg.get('seed', 42))

    sents, tags, vocab, tag2id, id2tag = prepare_data(cfg, use_toy=args.use_toy)
    max_len = cfg.get('max_len', 128)
    batch_size = cfg.get('batch_size', 32)
    epochs = cfg.get('epochs', 5)
    emb_dim = cfg.get('emb_dim', 100)
    hidden_dim = cfg.get('hidden_dim', 256)
    dropout = cfg.get('dropout', 0.5)
    lr = cfg.get('learning_rate', 1e-3)
    use_crf = cfg.get('use_crf', True)
    save_dir = cfg.get('save_dir', 'outputs/bilstm_crf')
    ensure_dir(save_dir)

    train_ds = ToyNERDataset(sents, tags, vocab, tag2id, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = BiLSTMCRF(vocab_size=len(vocab), tagset_size=len(tag2id), emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout, use_crf=use_crf)
    model.to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log_path = os.path.join(save_dir, 'train_log.tsv')
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write('epoch\tloss\tprecision\trecall\tf1\n')
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optim)
            p, r, f1 = eval_f1(model, train_loader, id2tag)
            lf.write(f'{epoch}\t{loss:.4f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\n')
            print(f"Epoch {epoch}: loss={loss:.4f} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    weight_path = os.path.join(save_dir, 'model.pt')
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'tag2id': tag2id, 'config': cfg}, weight_path)
    print(f'Model saved to {weight_path}')


if __name__ == "__main__":
    main()
