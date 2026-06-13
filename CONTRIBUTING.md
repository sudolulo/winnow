# Contributing to winnow

Bug reports, feature requests, and pull requests are all welcome.

## Before You Start

- Check [existing issues](https://github.com/sudolulo/winnow/issues) to avoid duplicates.
- For large changes, open an issue first to discuss the approach.
- All PRs target the `dev` branch — never `main` directly.

## Development Setup

Requires Python 3.13+ and [uv](https://astral.sh/uv).

```bash
git clone https://github.com/sudolulo/winnow.git
cd winnow
git checkout dev
uv sync
```

## Running Tests and Lint

```bash
uv run pytest          # run the test suite
uv run ruff check      # lint
uv run ruff check --fix  # auto-fix lint issues
```

CI runs both on every push and PR to `main` and `dev`. PRs must pass before merging.

## Pull Request Guidelines

- One logical change per PR.
- If you add behaviour, add a test for it.
- Keep the `CHANGELOG.md` entry in the `[Unreleased]` section updated.
- Commit messages should be plain English describing what changed and why.

## License

By submitting a contribution you agree that your work will be released under the project's [AGPLv3+ license](LICENSE).
