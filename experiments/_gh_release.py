"""创建 GitHub Release 并上传 wheel 资产.

凭据按顺序取: 环境变量 GH_TOKEN / GITHUB_TOKEN, 其次 ~/.config/gh/hosts.yml。
只读 notes 文件作为 release body, 不改动它。

用法:
  GH_TOKEN=xxx python -X utf8 experiments/_gh_release.py v0.3.0 docs/en/RELEASE_v0.3.0.md dist/*.whl
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "SunnyClaw-cx/SunnyNet"
API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
UA = "sunnynet-release/1.0"


def token() -> str:
    for k in ("GH_TOKEN", "GITHUB_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    hosts = Path.home() / ".config" / "gh" / "hosts.yml"
    if hosts.exists():
        for ln in hosts.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if s.startswith("oauth_token:"):
                return s.split(":", 1)[1].strip()
    raise SystemExit("未找到 GitHub 凭据 (设 GH_TOKEN 或写入 ~/.config/gh/hosts.yml)")


def call(method: str, url: str, tok: str, data=None, raw=None, ctype=None):
    body = raw if raw is not None else (
        json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", UA)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if ctype:
        req.add_header("Content-Type", ctype)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler())
    try:
        with opener.open(req, timeout=120) as r:
            txt = r.read().decode()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:600]}


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    tag, notes_path = sys.argv[1], sys.argv[2]
    assets = [Path(p) for p in sys.argv[3:]]
    tok = token()
    notes = Path(notes_path).read_text(encoding="utf-8")

    st, who = call("GET", f"{API}/user", tok)
    print(f"auth: {st} user={who.get('login', who)}")
    if st != 200:
        raise SystemExit(f"鉴权失败: {who}")

    st, body = call("POST", f"{API}/repos/{REPO}/releases", tok, data={
        "tag_name": tag,
        "name": tag,
        "body": notes,
        "draft": False,
        "prerelease": False,
    })
    if st == 422:
        print("release 已存在, 改取现有 release")
        st, rel = call("GET", f"{API}/repos/{REPO}/releases/tags/{tag}", tok)
        if st != 200:
            raise SystemExit(f"取 release 失败: {rel}")
    elif st not in (200, 201):
        raise SystemExit(f"创建 release 失败 {st}: {body}")
    else:
        rel = body
    print(f"release: {rel.get('html_url')} (id={rel.get('id')})")

    for a in assets:
        raw = a.read_bytes()
        ctype = ("application/octet-stream" if a.suffix == ".whl"
                 else "text/plain; charset=utf-8")
        st, res = call("POST",
                       f"{UPLOADS}/repos/{REPO}/releases/{rel['id']}/assets"
                       f"?name={a.name}", tok, raw=raw, ctype=ctype)
        if st in (200, 201):
            print(f"asset ok: {a.name} ({len(raw)} B) "
                  f"{res.get('browser_download_url', '')}")
        else:
            print(f"asset FAIL {a.name}: {st} {res}")
    print(f"DONE {rel.get('html_url')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())