# Contributing

1. Create a Python 3.10–3.12 virtual environment.
2. Install `python -m pip install -e '.[dev]'`.
3. Add tests for behavioral changes.
4. Run `./scripts/verify.sh` before opening a pull request.

Changes must preserve deterministic CPU execution, temporal split invariants, inspectable
artifacts, and the distinction between synthetic demonstrations and validated scientific results.
