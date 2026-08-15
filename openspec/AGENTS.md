# OpenSpec — ilma

This directory is the spec-driven development (SDD) source of truth for non-trivial
features in this repo. Specs live in `openspec/specs/`. One-folder-per-change proposals
live in `openspec/changes/`. Archived changes move to `openspec/changes/archive/`.

## For AI agents (Hermes, Claude, Cursor, etc.)

If you are working on a multi-file feature in this repo, follow the OpenSpec loop
defined in the `spec-driven-dev` skill (`~/.hermes/skills/spec-driven-dev/SKILL.md`):

1. `/opsx:explore <idea>` — think it through, no artifact yet.
2. `/opsx:propose <name> <why>` — create `openspec/changes/<name>/` with
   `proposal.md`, `specs/` (delta specs), `design.md`, `tasks.md`.
3. `/opsx:apply` — walk `tasks.md` top-to-bottom, one task per commit.
4. `/opsx:archive` — fold delta specs into `openspec/specs/`, move change folder
   to `openspec/changes/archive/<date>-<name>/`.

Skip this loop for one-line fixes and UI tweaks — those go straight to code.

## Conventions

- **Specs**: source of truth for *current* behavior. Organized by domain
  (`openspec/specs/<domain>/<capability>.md`).
- **Changes**: deltas to specs (`ADDED` / `MODIFIED` / `REMOVED`). Never rewrite
  the destination spec — describe the diff.
- **Tasks**: numbered, verb-first, each independently verifiable. One commit per task.
- **Archive**: history. Keep, don't delete.

## Tooling

```bash
openspec list              # list active changes
openspec list --specs      # list spec domains
openspec show <change>     # show a change's artifacts
openspec validate <change> --strict   # CI-friendly strict validation
openspec archive <change>  # archive a completed change
```

Installed via `npm install -g @fission-ai/openspec@latest`. Refresh after upgrades
with `openspec update`.
