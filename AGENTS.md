# Repository Guidelines

## Project Structure & Module Organization
- `ideal-buy-day.py` contains the full CLI workflow for downloading data, caching results, and producing day-of-month visualisations.
- `cache/` and `images/` are created at runtime; keep them out of version control unless a specific artifact is required for review.
- Place any future modules under `src/` if the codebase grows, and mirror that layout in `tests/` to keep imports straightforward.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment for the toolchain.
- `pip install -r requirements.txt` installs the pinned dependencies (currently `yfinance 0.2.66` and `matplotlib 3.9.2`); update this file when introducing new libraries or when upstream APIs change.
- `python ideal-buy-day.py AAPL -y 5` runs the non-interactive analysis for a ticker and saves a chart to `images/`.
- `python ideal-buy-day.py` launches the interactive prompt, useful for manual smoke-tests.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, lowercase `snake_case` for functions and variables, and descriptive argument names (`ticker_symbol`, not `sym`).
- Keep functions small and single-purpose as in the current refactors; add or update docstrings when behaviour changes.
- Prefer type hints for new helpers and format files with `black --line-length 100` before submitting.

## Testing Guidelines
- Add new tests under `tests/` using `pytest`; mirror module paths (e.g., `tests/test_analysis.py` for helpers in `analysis.py`).
- Use cached fixtures or vcrpy cassettes to avoid hitting the live Yahoo Finance API during test runs.
- Run `pytest -q` locally before opening a pull request and confirm generated charts still render with expected highlights.

## Commit & Pull Request Guidelines
- Recent history shows brief, action-focused messages (`Refactor. Break down into smaller purpose-specific functions.`); continue with concise, present-tense summaries.
- Reference related issues in the body, outline behavioural changes, and attach representative CLI output or image paths.
- PRs should describe validation steps (commands run, data sources used) and note whether caches or images need reviewers’ attention.

## Cache & Data Handling
- Cache files include the ticker, year span, and current date (`cache/AAPL_5_2024-05-12.pkl`); delete stale artifacts after major logic changes.
- Generated charts land in `images/` with a similar naming pattern; include the path when sharing results and avoid checking in large batches.
