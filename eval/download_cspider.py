"""CSpider 数据集下载助手。

CSpider 官方分发在网盘（Google Drive / 百度网盘），没有稳定的直链，
本脚本做两件事：
1. 尝试若干社区镜像直链（可用性随时间变化）；
2. 全部失败时打印手动下载指引（官网 + 期望的目录结构）。

期望解压后的目录结构：
  data/cspider/
  ├── train_spider.json
  ├── dev.json
  ├── tables.json
  └── database/<db_id>/<db_id>.sqlite   # 200 个库
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

CANDIDATES = {
    "dev.json": [
        # 社区镜像（按可用性维护；失效请提 PR 更新）
        "https://huggingface.co/datasets/xichitw/cspider/resolve/main/dev.json",
    ],
    "tables.json": [
        "https://huggingface.co/datasets/xichitw/cspider/resolve/main/tables.json",
    ],
    "train_spider.json": [
        "https://huggingface.co/datasets/xichitw/cspider/resolve/main/train_spider.json",
    ],
}

MANUAL_STEPS = """
[CSpider 手动下载指引]
1. 打开官方主页：https://taolusi.github.io/CSpider-explorer/
   （或配套仓库 https://github.com/taolusi/chisp 的 README）
2. 通过页面提供的 百度网盘 / Google Drive 链接下载 CSpider 1.0 数据包
3. 解压到 data/cspider/，并确认目录结构：
   data/cspider/train_spider.json
   data/cspider/dev.json
   data/cspider/tables.json
   data/cspider/database/<db_id>/<db_id>.sqlite
4. 运行：python -m eval.runner --split dev --limit 300 --label baseline

说明：在下载完成前，可先跑开箱即用的 mini 评测：
   python -m eval.mini_runner --label baseline
"""


def _try_download(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "datapilot/0.1"})
        with urlopen(req, timeout=30) as resp, open(dest, "wb") as f:  # noqa: S310
            f.write(resp.read())
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"  [skip] {url} -> {e}", file=sys.stderr)
        return False


def main() -> None:
    out_dir = Path("data/cspider")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for filename, urls in CANDIDATES.items():
        dest = out_dir / filename
        if dest.exists():
            print(f"[ok] {filename} 已存在")
            continue
        print(f"[..] 尝试下载 {filename}")
        ok = any(_try_download(url, dest) for url in urls)
        all_ok &= ok
        print(f"[{'ok' if ok else 'fail'}] {filename}")

    db_dir = out_dir / "database"
    if not db_dir.exists():
        all_ok = False

    if all_ok:
        print("\n数据就绪：python -m eval.runner --split dev --limit 300 --label baseline")
    else:
        print(MANUAL_STEPS)


if __name__ == "__main__":
    main()
