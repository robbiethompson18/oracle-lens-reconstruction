"""Build the WildChat oracle-lens dashboard: dashboard/template.html + readouts.jsonl (+ highlights.json) -> one HTML file."""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("readouts", default="runs/wildchat/readouts.jsonl", nargs="?")
ap.add_argument("-o", "--out", default="runs/wildchat/dashboard.html")
ap.add_argument("--highlights", default="dashboard/highlights.json")
args = ap.parse_args()

here = Path(__file__).parent
rows = [json.loads(line) for line in Path(args.readouts).read_text().splitlines() if line.strip()]
hl = json.loads(Path(args.highlights).read_text()) if Path(args.highlights).exists() else []
order = {h["id"]: i for i, h in enumerate(hl)}  # highlighted transcripts first, in highlight order
rows.sort(key=lambda r: order.get(r["id"], len(order)))
# per-token decoding splits multi-byte emoji into U+FFFD fragments; show them as a placeholder glyph
data = json.dumps({"transcripts": rows, "highlights": hl}, ensure_ascii=False).replace("</", "<\\/").replace("\ufffd", "\u25af")
html = (here / "template.html").read_text().replace("__DATA__", data)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(html)
print(f"{args.out}: {len(rows)} transcripts, {len(hl)} highlights, {len(html) / 1e6:.2f} MB")
