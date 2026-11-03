"""Merge a base Snake config with one or more overlays (later files win; nested dicts are merged key by key).

    python merge.py base.json boolean_head.json -o base_boolean.json
"""
import argparse
import json


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("overlays", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.base))
    for path in args.overlays:
        cfg = merge(cfg, json.load(open(path)))
    json.dump(cfg, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
