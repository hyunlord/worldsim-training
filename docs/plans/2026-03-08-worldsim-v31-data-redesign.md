# WorldSim Training Data v3.1 Redesign

## Scope

- Expand the training pipeline from `A-F` to `A-H`.
- Keep all assistant outputs as structured JSON.
- Add bilingual Korean/English fields wherever the task emits human-readable text.
- Add v3.1 context fields across the narrative/judgment tasks: `TEMP`, `STRESS`, `WORLD`.

## Task Map

- `A`: personality one-liner, `L4`
- `B`: situation reaction narrative, `L4`
- `C`: short dialogue, `L4`
- `D`: notification copy, `L4`
- `E`: action choice JSON, `L3`
- `F`: emotion judgment JSON, `L3`
- `G`: oracle interpretation JSON, `L5`
- `H`: worldbuilding text to `WorldRuleset` JSON IR, `L0`

## Input Contract

- Narrative and judgment tasks accept temperament context as TCI axes: `NS`, `HA`, `RD`, `P`.
- Stress is represented as a float in `0.0-1.0`.
- World context is a fixed tag from `default`, `dungeon`, `ocean`, `winter`, `magic`.
- Oracle interpretation adds `ORACLE`, `ROLE`, `RECENT`.
- Worldbuilding conversion adds `WORLDBUILDING`.

## Output Contract

- `A-F` keep their previous payload shape but now include temperament-derived metadata fields.
- `G` emits bilingual oracle interpretation plus action tendency, confidence, register, misinterpretation type, and temperament bias.
- `H` emits a fixed `WorldRuleset`-style JSON object with `resource_modifiers`, `special_zones`, `special_resources`, and `agent_modifiers`.
- `prepare_dataset.py` treats `L0`, `L3`, `L4`, and `L5` as first-class chat-training layers.

## Dataset Cardinality

- `A`: `10 personalities x 5 temperaments x 2 variants = 100`
- `B`: `10 situations x 8 emotions x 10 personalities x 2 variants = 1600`
- `C`: `10 situations x 10 personalities x 3 registers x 1 variant = 300`
- `D`: `10 situations x 5 variants = 50`
- `E`: `10 situations x 10 personalities x 2 variants = 200`
- `F`: `10 situations x 8 emotions x 10 personalities x 1 variant = 800`
- `G`: `20 oracles x 10 personalities x 4 temperaments x 1 variant = 800`
- `H`: `10 worldbuilding texts x 5 variants = 50`
- Task total: `3900`
- With negative and general samples, the planned mixed dataset total remains approximately `5400`.

## Validation Rules

- Korean narrative fields continue to use forbidden-word repair before pass/fail classification.
- Task-specific schema checks now cover `G` and `H`.
- `H` validates PascalCase names, English descriptions, array slot presence, and numeric ranges such as `multiplier`.
- Register checks still use the requested task register, not whatever the model self-reports.

## Runtime Notes

- `provider.require_parameters` stays `false` because the strict provider flag previously caused OpenRouter `404` routing failures.
- Task `H` can use a task-specific teacher model override.
- Structured response fallback remains available so generation can fall back from `json_schema` to `json_object` when a provider rejects the stricter format.
