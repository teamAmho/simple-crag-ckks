"""Combine per-dimension multi-query summaries into CSV and LaTeX outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_combined_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def scientific_latex(value: str) -> str:
    mantissa, exponent = f"{float(value):.3e}".split("e")
    return rf"${mantissa} \times 10^{{{int(exponent)}}}$"


def format_latex(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        body.append(
            " & ".join(
                [
                    row["dim"],
                    row["top_k"],
                    row["queries"],
                    f"{float(row['mean_topk_agreement']):.3f}",
                    f"{float(row['rank_agreement']):.3f}",
                    f"{float(row['full_rank_match_rate']):.3f}",
                    scientific_latex(row["mean_score_diff"]),
                    scientific_latex(row["max_score_diff"]),
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[hbt!]",
            r"\centering",
            r"\caption{Multi-query agreement between plaintext and encrypted retrieval.}",
            r"\label{tab:multi-query-accuracy}",
            r"\vspace{3pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{cccccccc}",
            r"\toprule",
            r"\textbf{Embedding dimension} & \textbf{Top-$k$} & \textbf{Queries} &",
            r"\textbf{Set agreement} & \textbf{Rank agreement} & \textbf{Full match} &",
            r"\textbf{Mean score error} & \textbf{Max score error} \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    args = parser.parse_args()

    rows = []
    for dim in (256, 512):
        rows.extend(read_rows(args.results_dir / f"multi_query_dim{dim}_summary.csv"))
    rows.sort(key=lambda row: (int(row["dim"]), int(row["top_k"])))

    combined_path = args.results_dir / "multi_query_summary.csv"
    latex_path = args.results_dir / "multi_query_table.tex"
    write_combined_csv(combined_path, rows)
    latex_path.write_text(format_latex(rows), encoding="utf-8")
    print(f"[save] {combined_path}")
    print(f"[save] {latex_path}")


if __name__ == "__main__":
    main()
