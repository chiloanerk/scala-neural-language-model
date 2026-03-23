# Contributing

Thanks for contributing to Scala Neural Language Model.

## Branch Strategy

Do not commit directly to `master`. Open all changes through Pull Requests.

Use one focused branch per change:

- `feature/<short-kebab-case>` for new features
- `bugfix/<short-kebab-case>` for bug fixes
- `docs/<short-kebab-case>` for documentation-only changes
- `chore/<short-kebab-case>` for maintenance/refactors
- `release/<version>` for release preparation

Examples:

- `feature/interactive-run-launcher`
- `bugfix/interrupt-auto-resume-cancel`
- `docs/readme-quickstart-refresh`

## Pull Request Guidelines

- Keep PRs focused (one concern per PR).
- Use clear titles, for example:
  - `feat: add interactive benchmark wizard`
  - `fix: clear interrupt snapshot when resume is canceled`
  - `docs: refresh quick start examples`
- Include these sections in PR description:
  - Summary
  - What changed
  - Why
  - Test evidence

## Local Validation Before PR

Run at least:

```bash
sbt -batch test
```

For focused updates, run relevant suites such as:

```bash
sbt -batch "testOnly app.MainConfigSuite"
```

## Merge Policy

- Prefer squash merge for a clean `master` history.
- Delete merged branches.
- Keep commit messages concise and intent-focused.

