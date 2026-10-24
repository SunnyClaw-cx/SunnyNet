"""Widen the code_python corpus with real library source (numpy/flask/requests/urllib3).

Supplements prepare_data's TheAlgorithms fetch. Each repo's .py files are decoded,
boilerplate-stripped, and saved under code_python/ so the mix share is meaningful.
Tolerant: any repo that fails to download is skipped, never fatal.
"""

from __future__ import annotations

import io
import tarfile
import urllib.error
import urllib.request

import prepare_data as pd

UA = {"User-Agent": "Mozilla/5.0 (sunnynet-corpus)"}
REPOS = [
    ("numpy/numpy", "main"),
    ("pallets/flask", "main"),
    ("psf/requests", "main"),
    ("urllib3/urllib3", "main"),
]
MIN_CHARS, MAX_BYTES = 500, 300_000


def fetch_repo(repo: str, branch: str) -> int:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    try:
        data = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=180
        ).read()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [skip] {repo}: {e}")
        return 0
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".py")):
                continue
            if not (200 < m.size < MAX_BYTES):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            text = pd.decode_any(fh.read())
            if not text:
                continue
            body = pd.clean_code(text)
            if len(body) < MIN_CHARS:
                continue
            name = f"repo_{repo.replace('/', '+')}__{m.name.replace('/', '__')}"
            pd.save("code_python", name, body)
            n += 1
    return n


def main() -> int:
    total = 0
    for repo, branch in REPOS:
        n = fetch_repo(repo, branch)
        total += n
        print(f"  {repo}: {n} files")
    print(f"  total: {total} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
