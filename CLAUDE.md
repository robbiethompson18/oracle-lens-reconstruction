# oracle-lens-reconstruction

Oracle lens (activation -> reconstructing phrases) for Qwen3.6-27B, replicating Anthropic's
global-workspace appendix (Gurnee et al. 2026) and ceselder's July 2026 port.

@~/.claude/personal-repo-rules.md

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

## Stack

- Python: `uv` for deps/venv, `ruff` for lint + format (line length 140), `ty` for types. Run
  `uv run ruff check . && uv run ruff format . && uv run ty check` before shipping.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a
  test suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.
- GPU work runs on a RunPod pod via SkyPilot (`sky/pod.yaml`); code is shipped over ssh with
  `sky/ship.sh`. The local Mac never loads the model. `src/olens` is the pipeline; each stage is
  `python -m olens.<stage> --config configs/<name>.yaml`.

## Docs

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines) → `docs/*.md`, with a one-line index entry below.
- **Local-only docs** (not in git) → `docs/local/*.md`.

See `~/.claude/personal-repo-rules.md` (imported above) for the full convention.

Current docs:
- [Plan — read first; lineages, decisions, recipe, costs, uncertainties](docs/plan.md)
