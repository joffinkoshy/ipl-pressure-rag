"""
End-to-end runner — reproduces every artefact from scratch, in order.

    python run_all.py

Stages (each is independently runnable too):
    1. main.py                 raw Cricsheet CSVs  -> data/ipl_2024_25_26.csv
    2. wikipedia_fetcher.py    Wikipedia seasons/teams + match_importance.json
    3. pressure_index.py       Pressure Index      -> ipl_with_pressure_index.csv (+plot)
    4. rag.run_questions       5 multi-hop answers -> answers.md (fetches players)
    5. rag.break_pipeline      Part 3 precision    -> part3_break_pipeline.md
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STAGES = [
    ("Load Cricsheet (3 seasons)", [PY, "main.py"]),
    ("Fetch Wikipedia + match importance", [PY, "wikipedia_fetcher.py"]),
    ("Compute Pressure Index", [PY, "pressure_index.py"]),
    ("Answer 5 questions (multi-hop RAG)", [PY, "-m", "rag.run_questions"]),
    ("Part 3 — break the pipeline", [PY, "-m", "rag.break_pipeline"]),
]


def main():
    for i, (label, cmd) in enumerate(STAGES, 1):
        print(f"\n{'#'*72}\n# STAGE {i}/{len(STAGES)}: {label}\n{'#'*72}")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"\n!! Stage {i} failed (exit {r.returncode}). Stopping.")
            sys.exit(r.returncode)
    print("\nAll stages complete. See data/outputs/ for results.")


if __name__ == '__main__':
    main()
