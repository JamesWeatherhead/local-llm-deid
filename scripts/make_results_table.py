#!/usr/bin/env python3
"""Build the ranked per-model headline table from ``metrics_long.csv``.

This reconstructs the manuscript's main results table directly from the scorer
output. It reads the note-wide (``phi_type == ALL``) rows for one pass and
cohort, ranks models by character recall, and prints an aligned table; with
``--csv`` it also writes the selected rows to a file.

By default it uses the headline configuration reported in the paper -- the
cumulative two-pass predictions over every expected note::

    python scripts/make_results_table.py --metrics out/metrics_long.csv

Switch pass or cohort to inspect the other views the scorer emits::

    python scripts/make_results_table.py --metrics out/metrics_long.csv \\
        --pass pass1 --cohort operational_complete

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

# Columns shown, in order: (csv field, header, kind). "num" is a rate rendered
# to three decimals; "count" is an integer; "raw" is printed verbatim.
DISPLAY = [
    ("params_total_B", "Params(B)", "raw"),
    ("is_medical", "Med", "flag"),
    ("char_recall", "Recall", "num"),
    ("char_precision", "Prec", "num"),
    ("char_f1", "F1", "num"),
    ("strict_span_recall", "StrictR", "num"),
    ("relaxed_span_recall", "RelaxR", "num"),
    ("zero_residual_note_rate", "CleanNote", "num"),
    ("notes_with_leak_rate", "LeakNote", "num"),
]


def fmt(value: str, kind: str) -> str:
    """Render one cell; blank ('undefined') becomes a dash."""
    if value in ("", "None"):
        return "-"
    if kind == "num":
        return f"{float(value):.3f}"
    if kind == "flag":
        return {"True": "yes", "False": "", "1": "yes", "0": ""}.get(value, value)
    return value


def select_rows(metrics_path: Path, pass_label: str, cohort: str) -> list[dict[str, str]]:
    """Note-wide rows for one pass/cohort, ranked by descending character recall."""
    with metrics_path.open(encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["phi_type"] == "ALL" and row["pass"] == pass_label and row["cohort"] == cohort
        ]
    rows.sort(key=lambda r: float(r["char_recall"]) if r["char_recall"] not in ("", "None") else -1.0,
              reverse=True)
    return rows


def render_table(rows: list[dict[str, str]]) -> str:
    """Format the ranked rows as a fixed-width text table."""
    headers = ["#", "model_id"] + [head for _, head, _ in DISPLAY]
    lines = [[str(i)] + [row["model_id"]] + [fmt(row.get(field, ""), kind)
                                             for field, _, kind in DISPLAY]
             for i, row in enumerate(rows, start=1)]

    widths = [len(h) for h in headers]
    for line in lines:
        widths = [max(w, len(cell)) for w, cell in zip(widths, line)]

    def join(cells: list[str]) -> str:
        # First two columns left-aligned (rank, name); numeric columns right.
        out = [cells[0].ljust(widths[0]), cells[1].ljust(widths[1])]
        out += [cell.rjust(w) for cell, w in zip(cells[2:], widths[2:])]
        return "  ".join(out)

    body = [join(headers), join(["-" * w for w in widths])]
    body += [join(line) for line in lines]
    return "\n".join(body)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["rank", "model_id"] + [field for field, _, _ in DISPLAY]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, "model_id": row["model_id"],
                             **{field: row.get(field, "") for field, _, _ in DISPLAY}})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics", type=Path, default=Path("out/metrics_long.csv"),
                        help="path to metrics_long.csv (default: out/metrics_long.csv)")
    parser.add_argument("--pass", dest="pass_label", default="cumulative_pass2",
                        choices=["pass1", "cumulative_pass2"],
                        help="which pass to report (default: cumulative_pass2)")
    parser.add_argument("--cohort", default="all_expected",
                        choices=["all_expected", "operational_complete"],
                        help="which cohort to report (default: all_expected)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="optional path to also write the ranked table as CSV")
    args = parser.parse_args(argv)

    rows = select_rows(args.metrics, args.pass_label, args.cohort)
    if not rows:
        parser.error(f"no ALL rows for pass={args.pass_label} cohort={args.cohort} in {args.metrics}")

    print(f"# {args.pass_label} / {args.cohort} -- {len(rows)} models, ranked by character recall\n")
    print(render_table(rows))
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nwrote {len(rows)} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
