"""
Build report/report.pdf from report/report.md.

Renders the Markdown to HTML and prints it to PDF with a headless Chromium
browser (Microsoft Edge or Google Chrome).

Usage:
    pip install markdown
    python report/build_pdf.py
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

REPORT_DIR = Path(__file__).resolve().parent
SOURCE = REPORT_DIR / "report.md"
OUTPUT = REPORT_DIR / "report.pdf"

BROWSERS = [
    "msedge",
    "microsoft-edge",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]

NAVY = "#1f3864"
CSS = f"""
@page {{ size: Letter; margin: 0.9in 1in; }}
body {{ font-family: Calibri, Carlito, "Segoe UI", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.33; color: #1a1a1a; }}
h1, h2, h3 {{ color: {NAVY}; line-height: 1.2; break-after: avoid; }}
h2 {{ font-size: 15pt; margin: 18pt 0 5pt; }}
h3 {{ font-size: 12pt; margin: 12pt 0 3pt; }}
p, li {{ text-align: left; }}
ul, ol {{ padding-left: 20pt; }}
li {{ margin: 3pt 0; }}
a {{ color: #2e5da8; text-decoration: none; }}
code {{ font-family: Consolas, "Courier New", monospace; font-size: 9.3pt; }}

.title-page {{ text-align: center; break-after: page; padding-top: 1.7in; }}
.title-page h1 {{ font-size: 24pt; margin: 0 0 10pt; }}
.title-page h2 {{ font-size: 15pt; margin: 0 0 6pt; }}
.title-page p {{ text-align: center; margin: 10pt 0; }}
.title-page > p:first-of-type {{ color: #555; margin-bottom: 36pt; }}
.title-page blockquote {{ margin-top: 48pt; }}

blockquote {{ margin: 12pt 0; padding: 9pt 12pt; background: #eef2f8;
             border-left: 4pt solid {NAVY}; font-size: 9.5pt; text-align: left; break-inside: avoid; }}
blockquote p {{ margin: 0; text-align: left !important; }}

table {{ width: 100%; border-collapse: collapse; margin: 8pt 0 10pt; font-size: 9pt; break-inside: avoid; }}
th {{ background: {NAVY}; color: #fff; font-weight: bold; white-space: nowrap; }}
td:first-child {{ white-space: nowrap; }}
th, td {{ border: 0.6pt solid #b7bfcc; padding: 4pt 6pt; vertical-align: top; }}
tr:nth-child(even) td {{ background: #f4f6fa; }}
tr {{ break-inside: avoid; }}

p:has(> img) {{ text-align: center; margin: 10pt 0 2pt; break-inside: avoid; break-after: avoid; }}
img {{ max-width: 100%; max-height: 3.6in; }}
p:has(> img) + p > em:only-child {{ display: block; font-size: 9pt; color: #444; }}
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        path = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if path:
            return path
    sys.exit("No Microsoft Edge or Google Chrome found. Install one to build the PDF.")


def wait_for_pdf(path: Path, timeout: float = 60) -> None:
    """Edge on Windows returns before the PDF is written, so wait until its size settles."""
    deadline = time.monotonic() + timeout
    last_size = -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else -1
        if size > 0 and size == last_size:
            return
        last_size = size
        time.sleep(1)
    sys.exit(f"Timed out waiting for the browser to write {path}")


def main() -> None:
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "md_in_html", "sane_lists"],
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>CSE437 Final Project Report</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    # Write the HTML next to report.md so relative image paths (../figures/...) resolve
    page = REPORT_DIR / "_build.html"
    page.write_text(html, encoding="utf-8")
    OUTPUT.unlink(missing_ok=True)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
            subprocess.run(
                [
                    find_browser(),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    f"--user-data-dir={profile}",
                    "--no-pdf-header-footer",
                    f"--print-to-pdf={OUTPUT}",
                    page.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            wait_for_pdf(OUTPUT)
    finally:
        page.unlink()
    print(f"Wrote {OUTPUT.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
