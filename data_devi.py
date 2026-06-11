# split_office31_txt.py
import argparse, random, os
from collections import defaultdict

def read_list(txt_path):
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # 末尾のラベルだけを安全に切り出す（パスに空白があってもOK）
            path, label = ln.rsplit(" ", 1)
            lines.append((path, int(label)))
    return lines

def ensure_leading_slash(path, ensure=True):
    if not ensure:
        return path
    # 先頭に / が無ければ付与（preprocessコードが args.input_dir + 相対パス を前提にしているため）
    if not path.startswith("/") and not path.startswith("\\"):
        return "/" + path.replace("\\", "/")
    return path.replace("\\", "/")

def split_stratified(pairs, test_ratio, seed=1234):
    """pairs: List[(path, label)] をラベルごとに分割"""
    random.seed(seed)
    buckets = defaultdict(list)
    for p, y in pairs:
        buckets[y].append((p, y))

    train, test = [], []
    for y, items in buckets.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_ratio)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    random.shuffle(train)
    random.shuffle(test)
    return train, test

def write_txt(out_path, pairs, add_leading_slash=True):
    with open(out_path, "w", encoding="utf-8") as f:
        for p, y in pairs:
            f.write(f"{ensure_leading_slash(p, add_leading_slash)} {y}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_txt", required=True, help="全画像リスト（例: amazon_all.txt）")
    ap.add_argument("--outdir", required=True, help="出力先ディレクトリ（例: data/office）")
    ap.add_argument("--domain", required=True, help="ドメイン名（例: amazon）")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="testの割合（例: 0.2）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--leading_slash", type=int, default=1, help="先頭に/を付ける(1) or 付けない(0)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    pairs = read_list(args.in_txt)
    train, test = split_stratified(pairs, args.test_ratio, seed=args.seed)

    train_out = os.path.join(args.outdir, f"{args.domain}_train.txt")
    test_out  = os.path.join(args.outdir, f"{args.domain}_test.txt")
    write_txt(train_out, train, add_leading_slash=bool(args.leading_slash))
    write_txt(test_out,  test,  add_leading_slash=bool(args.leading_slash))

    print(f"[{args.domain}] train: {len(train)}  test: {len(test)}  (classes≈{len(set(y for _,y in pairs))})")
    print(f"Wrote: {train_out}")
    print(f"Wrote: {test_out}")

if __name__ == "__main__":
    main()
