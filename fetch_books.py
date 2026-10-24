"""Diverse public-domain English book corpus via the gutendex API.

Supplements prepare_data's fixed 13 classics with the most-read English books
on Project Gutenberg (ranked by download_count), so the EN training corpus is
large and *diverse* instead of a handful of repeated novels. Reuses
prepare_data's fetch/decode/clean/save so output is byte-compatible.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import prepare_data as pd

UA = {"User-Agent": "Mozilla/5.0 (sunnynet-corpus)"}
GUTENDEX = "https://gutendex.com/books/?languages=en&per_page={per_page}&page={page}"


def gutendex_books(per_page: int = 100, pages: int = 5) -> list[dict]:
    out: list[dict] = []
    for p in range(1, pages + 1):
        url = GUTENDEX.format(per_page=per_page, page=p)
        try:
            d = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=90
                ).read()
            )
            out += d.get("results", [])
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            # keep going - one slow page shouldn't sink the run
            print(f"  [page {p}] ERR {e}")
    return out


def grab(b: dict) -> tuple[int, str] | None:
    url = b.get("formats", {}).get("text/plain; charset=utf-8")
    if not url:
        return None
    data = pd.fetch(url, timeout=90)
    if not data or len(data) < 2000:
        return None
    t = pd.decode_any(data)
    return (b["id"], t) if t else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--top", type=int, default=300, help="取 download_count 最高的 N 本"
    )
    ap.add_argument("--pages", type=int, default=5, help="gutendex 翻页数 (每页 100)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print(f"[gutendex] 抓取 EN 书目 (pages={args.pages}) ...")
    books = gutendex_books(pages=args.pages)
    books = [b for b in books if not b.get("copyright")]
    books.sort(key=lambda b: b.get("download_count", 0), reverse=True)
    top = books[: args.top]
    print(f"  候选 {len(books)} 本公开领域英文书, 取下载量最高的 {len(top)} 本")

    stats = {"zh_ancient": 0, "en_literature": 0, "drop": 0}
    chars = {"zh_ancient": 0, "en_literature": 0}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(grab, b): b for b in top}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r is None:
                stats["drop"] += 1
            else:
                bid, raw = r
                body = pd.strip_boilerplate(raw)
                dom = pd.classify(body)
                if dom == "drop":
                    stats["drop"] += 1
                    continue
                body = pd.clean_zh(body) if dom == "zh_ancient" else pd.clean_en(body)
                if len(body) < 8000:
                    stats["drop"] += 1
                    continue
                chars[dom] += pd.save(dom, f"gutendex_{bid}.txt", body)
                stats[dom] += 1
            if i % 50 == 0:
                print(
                    f"  ...{i}/{len(top)}  en={stats['en_literature']} drop={stats['drop']}"
                )

    print("  分域结果:", stats)
    print("  字符数:", {k: f"{v:,}" for k, v in chars.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
