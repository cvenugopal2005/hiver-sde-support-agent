# Decision Log

A plain list of non-obvious decisions made during this project, and why.

1. **Project structure** — Used a standard `src/`-layout Python package
   (`src/hiver_agent/`) with separate top-level directories for data, notebooks,
   tests, scripts, configs, evaluations, reports, and docs. This keeps runtime
   code, experiments, evaluation artifacts, and deliverables (report, decision
   log) clearly separated, which matches the deliverables the assignment asks
   for (repo, golden eval set, evaluation harness, report, decision log) without
   introducing any extra infrastructure.

2. **Python version** — Targeting Python 3.11. It's a stable, widely-available
   version with good library support for the kind of data/LLM work this
   assignment requires, and doesn't require justification beyond "a recent,
   stable interpreter."
