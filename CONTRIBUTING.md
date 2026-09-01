# Contributing to Enhanced World Model

Thanks for your interest in contributing! This guide covers what you need to know before opening a PR.

## Setup

```bash
git clone https://github.com/Larwive/Enhanced-World-Model.git
cd Enhanced-World-Model
uv sync --extra dev          # or: pip install -e ".[dev]"
uv run pre-commit install    # required
```

Pre-commit runs ruff (lint/format) and mypy automatically on every commit.

## Workflow

1. **Find or open an issue** before starting significant work.
2. **Branch from `main`**: `<type>/GH-<issue>/<short-description>`
   Types: `feature`, `fix`, `docs`, `refactor`, `test`, `chore`
   Example: `git checkout -b feature/GH-002/add-temporal-transformer`
3. **Make focused changes**, target one issue per PR. Split larger work into multiple PRs.
4. **Add tests** for new functionality.
5. **Run checks before pushing**:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/
   ```
6. **Push and open a PR**, linking the issue with `Closes #XXX`.

## Commits

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>
```

- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `perf`
- Scopes: `vision`, `memory`, `controller`, `world-model`, `training`, `deps`, `ci`, `tests`
- Imperative mood, lowercase, no trailing period, under 72 chars

```bash
feat(vision): add VQ-VAE with EMA quantizer
fix(memory): handle dynamic batch size in sequence buffer
refactor(controller): separate discrete and continuous implementations
```

For breaking changes, add `!` after the scope and a `BREAKING CHANGE:` footer explaining the impact.

## Pull Requests

- **Title**: `[GH-XXX] type(scope): brief description`
- PR requires **1 approval** and passing CI. We use **squash and merge**.
- Reviewers check: single issue scope, tests included, mypy/ruff pass, docs updated if needed.

## Code Quality

- **Type hints on every function**: mypy runs in normal mode (see `pyproject.toml` for config).
- **Formatting/linting** is automated via Ruff (`uv run ruff format src/`, `uv run ruff check --fix src/`) — 100-char lines, double quotes, 4-space indent.
- **Docstrings**: Google style, required for public functions/classes.
- **Import order**: stdlib → third-party → local, each group separated by a blank line.

## Testing

- Unit tests for new components, integration tests for cross-component behavior.
- Run with `uv run pytest tests/` (add `-v` or `--cov=src` as needed).
- Mirror the module structure in `tests/` (e.g. `src/vision/` → `tests/test_vision.py`).

## Questions?

Check existing issues/PRs and `ARCHITECTURE_DECISIONS.md` first, then open an issue with the `question` label.

## License

By contributing, you agree your contributions are licensed under the project's license.
