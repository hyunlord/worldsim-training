# Data Directory Rules

- Treat `raw/`, `validated/`, and `final/` as generated outputs. Do not hand-edit files there.
- Promotion flow is fixed: `raw -> validated -> final`.
- Curated fixtures belong in `samples/` and may stay in git.
- Keep large generated files out of git unless there is an explicit archival reason.
