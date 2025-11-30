import os
import random
import argparse
import yaml
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Toy dataset (same as CRF file)
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
# Attention module
# -----------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** 0.5

    def forward(self, Q, K, V, mask=None):
        # Q,K,V: [B,T,D]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if mask is not None:
            # mask: [B,T] -> expand to [B,1,T] then broadcast
            attn_mask = (mask == 0).unsqueeze(1)
            scores = scores.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        return out, attn


# -----------------------
# BiLSTM-Attention Model
# -----------------------
class BiLSTMAttn(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=100, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.attn = ScaledDotProductAttention(hidden_dim)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x, mask):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        Q = self.proj_q(out)
        K = self.proj_k(out)
        V = self.proj_v(out)
        ctx, attn = self.attn(Q, K, V, mask)
        h = self.dropout(ctx)
        logits = self.fc(h)
        return logits, attn


# -----------------------
# Training & Eval
# -----------------------

def train_one_epoch(model, loader, optim):
    model.train()
    total = 0.0
    for ids, tags, mask in loader:
        ids, tags, mask = ids.to(dev), tags.to(dev), mask.to(dev)
        logits, _ = model(ids, mask)
        B, T, C = logits.shape
        loss = nn.CrossEntropyLoss()(logits.view(B*T, C)[mask.view(B*T).bool()], tags.view(B*T)[mask.view(B*T).bool()])
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
            logits, attn = model(ids, mask)
            pred = logits.argmax(-1)
            B, T = pred.shape
            for b in range(B):
                gold = tags[b][: mask[b].sum()].cpu().tolist()
                pr = pred[b][: mask[b].sum()].cpu().tolist()
                for g, p in zip(gold, pr):
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
    ap.add_argument('--config', type=str, default='config/bilstm_attn.yaml')
    ap.add_argument('--use_toy', action='store_true')
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
    save_dir = cfg.get('save_dir', 'outputs/bilstm_attn')
    ensure_dir(save_dir)

    ds = ToyNERDataset(sents, tags, vocab, tag2id, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BiLSTMAttn(vocab_size=len(vocab), tagset_size=len(tag2id), emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    model.to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log_path = os.path.join(save_dir, 'train_log.tsv')
    with open(log_path, 'w', encoding='utf-8') as lf:
        lf.write('epoch\tloss\tprecision\trecall\tf1\n')
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, loader, optim)
            p, r, f1 = eval_f1(model, loader, id2tag)
            lf.write(f'{epoch}\t{loss:.4f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\n')
            print(f"Epoch {epoch}: loss={loss:.4f} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    weight_path = os.path.join(save_dir, 'model.pt')
    torch.save({'state_dict': model.state_dict(), 'vocab': vocab, 'tag2id': tag2id, 'config': cfg}, weight_path)
    print(f'Model saved to {weight_path}')


if __name__ == "__main__":
    main()
