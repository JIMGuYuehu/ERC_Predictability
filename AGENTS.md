# Code Update Git Policy

- After every code, notebook, or script update, run a lightweight sanity check when practical.
- Before the final response, commit the relevant changes and push them to the configured Git remote so the work can be undone from repository history.
- If the current directory is not a valid Git repository, or if pushing is unavailable, state that clearly in the final response and list the changed files.
- Do not include unrelated user changes in commits. Inspect the working tree first and commit only the files touched for the current request.
- Do not use destructive Git commands such as `git reset --hard` or checkout-based reverts unless the user explicitly asks for them.

# Repository Scope Policy

- Treat `/home/weiji/restart_exam/code_cleaned` as the only writable project scope for Codex work.
- Do not create, edit, delete, format, move, commit, or otherwise modify files outside `/home/weiji/restart_exam/code_cleaned` unless the user explicitly gives a newer instruction for that specific operation.
- Prefer running commands from `/home/weiji/restart_exam/code_cleaned` so git operations and generated files stay inside this repository.

# GitHub Traceability Policy

- Every Codex-made change must be committed and pushed to the configured GitHub remote before the final response whenever pushing is available.
- Use focused commits that contain only files relevant to the current request.
- Run `git status --short` before staging, after committing, and after pushing.
- If GitHub push is blocked by credentials, network, or remote configuration, report the blocker clearly and list the changed files.

# Runtime Environment Policy

- Run project code with the `jimnew` environment.
- Prefer `/home/weiji/miniconda3/envs/jimnew/bin/python` for Python commands.
- For tools installed only through Conda, use the `jimnew` environment equivalent rather than the system environment.

# Repo-Local Codex Skill

- Treat this Markdown file as the persistent repo-local Codex skill for `code_cleaned`.
- Re-read and follow these rules before making any future code, notebook, script, or documentation change in this repository.
