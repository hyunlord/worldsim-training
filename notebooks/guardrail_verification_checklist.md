# Guardrail Verification Checklist

Use this checklist during the DGX Spark baseline run that validates the guardrail stack merged in `e3aa0b6`.

## 1. Pre-Run Validation

- [ ] `git log -1 --oneline` shows commit `e3aa0b6` or newer on `main`
- [ ] `python -c "from training.lib.json_repair import repair_json; print('OK')"` passes
- [ ] `python -c "from training.lib.json_sanitize import sanitize_keys; print('OK')"` passes
- [ ] `python -c "from training.lib.structured_metrics import BatchMetrics; print('OK')"` passes
- [ ] CUDA is visible from the active notebook kernel
- [ ] `bitsandbytes`, `transformers`, `peft`, `datasets`, `accelerate`, and `trl` import correctly in the notebook kernel
- [ ] Dataset paths still resolve:
  - `data/training/worldsim-v31-mix-v1/train_converted.jsonl`
  - `data/training/worldsim-v31-mix-v1/dev_converted.jsonl`

## 2. Notebook Execution

- [ ] Open `notebooks/dgx_spark_qlora_train.ipynb`
- [ ] Run repo-root guard cell
- [ ] Run environment visibility cell
- [ ] Run strict true-QLoRA preflight cell
- [ ] Confirm `preflight["ok"]` is `True`
- [ ] Review baseline config before launch
- [ ] Run dataset preview cell
- [ ] Run trainer invocation cell
- [ ] Wait for training to complete successfully
- [ ] Run post-run artifact cells
- [ ] Run analyzer + registry section
- [ ] Run the new `GUARDRAIL IMPACT SUMMARY` cell
- [ ] Run final operational judgment cell

## 3. Post-Run Metrics Collection

- [ ] Identify the completed baseline output directory under `outputs/baseline/worldsim-v31-mix-v1/<run_id>`
- [ ] Confirm these files exist:
  - `run_config.json`
  - `summary.json`
  - `metrics.json`
  - `sample_generations.jsonl`
  - `analysis_report.json`
- [ ] Run:

```bash
python scripts/extract_guardrail_metrics.py outputs/baseline/worldsim-v31-mix-v1/<run_id>
```

- [ ] Confirm `guardrail_verification_report.json` is written into the same output directory

## 4. Comparison Table

Fill this in after the DGX run completes.

| Metric | Pre-Guardrail | Post-Guardrail | Delta |
| --- | ---: | ---: | ---: |
| structured_success_rate | ~0.70-0.85 |  |  |
| json_parse_failure_rate | ~0.10-0.20 |  |  |
| repair_applied_rate | N/A |  |  |
| extra_key_rate | N/A |  |  |
| retry_rate | N/A |  |  |
| verdict | FAIL_ARTIFACT_INVALID |  |  |

## 5. Decision Gate

- `PASS`
  - `structured_success_rate >= 0.95`
  - Next step: proceed with baseline quality analysis and semantic tuning only

- `PARTIAL`
  - `0.85 <= structured_success_rate < 0.95`
  - Next step: prompt contract hardening or constrained decoding follow-up

- `INSUFFICIENT`
  - `structured_success_rate < 0.85`
  - Next step: constrained decoding and/or data quality intervention required

## 6. Run Artifacts to Attach Back Into Review

- [ ] `summary.json`
- [ ] `metrics.json`
- [ ] `analysis_report.json`
- [ ] `guardrail_verification_report.json`
- [ ] one short written note on whether remaining failures are:
  - mostly repaired by guardrails
  - still true model failures
  - concentrated in specific tasks
