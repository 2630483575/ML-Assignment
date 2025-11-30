import os
from typing import List, Tuple

DEF_TAG = "O"

def read_conll(path: str, token_col: int = 0, tag_col: int = -1) -> Tuple[List[List[str]], List[List[str]]]:
    """Read CoNLL-like file.

    Each non-empty line: token [col2 ...] tag
    Blank line separates sentences.
    token_col: index of token column
    tag_col: index of tag column (supports negative index)
    """
    sentences: List[List[str]] = []
    tags: List[List[str]] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_tokens:
                    sentences.append(cur_tokens)
                    tags.append(cur_tags)
                    cur_tokens, cur_tags = [], []
                continue
            parts = line.split()
            if len(parts) <= max(token_col, tag_col if tag_col >=0 else len(parts)+tag_col):
                # skip malformed line
                continue
            tok = parts[token_col]
            tg = parts[tag_col]
            cur_tokens.append(tok)
            cur_tags.append(tg)
    if cur_tokens:
        sentences.append(cur_tokens)
        tags.append(cur_tags)
    return sentences, tags


def build_vocab(sentences: List[List[str]]) -> dict:
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in sentences:
        for w in s:
            lw = w.lower()
            if lw not in vocab:
                vocab[lw] = len(vocab)
    return vocab


def build_tagset(tags: List[List[str]]) -> Tuple[dict, dict]:
    all_tags = set()
    for seq in tags:
        for t in seq:
            all_tags.add(t)
    if DEF_TAG not in all_tags:
        all_tags.add(DEF_TAG)
    sorted_tags = sorted(all_tags)
    tag2id = {t: i for i, t in enumerate(sorted_tags)}
    id2tag = {i: t for t, i in tag2id.items()}
    return tag2id, id2tag


def try_read(path: str):
    try:
        return read_conll(path)
    except Exception:
        return None
