# Working on this repo

Ground rules for any AI agent (Claude, Codex, or otherwise) making changes
here. They exist because of a specific incident: Claude and Codex each built
substantial, overlapping features on separate branches for several days
without merging, and reconciling them required a manual, line-by-line merge
of two divergent rewrites of the same core files. Everything below is aimed
at not repeating that.

## Ship straight to main

When asked for a change: implement it, run the checks below, commit, and
merge to `main` directly. Don't leave finished work stranded on an unmerged
branch, and don't open a pull request for review unless asked to.

**Exception:** destructive git operations - force-push, rewriting history,
deleting a branch, discarding uncommitted work - always get confirmed with
the user first, regardless of the above.

## Don't diverge for long

The root cause of the incident above was structural, not a mistake either
agent made: two agents had real, substantial tasks on separate branches, and
neither had a reason to notice the other's changes until both had grown
large. The longer two branches diverge before meeting, the more expensive and
riskier the reunion.

- Before starting substantial new work, pull the latest `main`. If another
  agent's work is sitting unmerged on a branch, land it - or find out why it
  isn't landed - before building something new on top of the old `main`.
- Prefer small, frequently-merged changes over long-lived feature branches.
- If you must work on a branch for more than a session, merge `main` into it
  (or rebase onto it) regularly rather than only at the end.

## Before merging anything

- Run `pytest tests -q` and `pyflakes app tests scripts`. Both must be clean.
- Check that CI (GitHub Actions) is green on the pushed commit before
  considering the work done - don't rely on local results alone.

## Renames and other API changes

Grep the *whole* tree (`app/`, `tests/`, `docs/`, `static/`) for every call
site before considering a rename done. A stale call site can pass a text
merge cleanly and only fail at runtime - that has already happened once in
this repo (`VideoRepository.set_keypoints_path` -> `set_latest_keypoints_path`,
missed in a follow-up branch that was written before the rename landed).

## Keep the docs honest

If behavior changes, check `README.md` and `docs/*.md` for lines that are now
false. This repo's docs have gone stale silently more than once already: a
strapline claiming "no smoothing" next to a shipped smoothing toggle, an API
section describing synchronous upload after it became async, endpoints and
config variables that existed in code but never made it into the docs. When
you finish a change, diff the docs against the code, not just against your
memory of what you wrote.

## Cross-platform

This app is self-hosted on both Windows and Linux (see `docs/SELF_HOST.md`).
Don't write install, path, or service commands in one OS's dialect only -
give the Windows and Linux/macOS forms side by side, the way the tunnel
section of that doc does.

## Scope discipline

A requested fix is not license to also change unrelated behavior without
calling it out. If fixing a flaky test, or resolving a merge conflict, turns
up a real behavior change along the way (a boundary condition, a retry limit,
a default), make that change visible and separate rather than folding it
silently into the commit it was found in.

## Where things are

- `README.md` - what the app does and why, the load-bearing design decisions.
- `docs/SELF_HOST.md` / `docs/DEPLOY.md` - running it on your own machine or
  on Fly.io.
- `tests/` - run the suite before every merge; `pytest tests -q` prints the
  current count, so this file doesn't need to.
