# NLSeeker evaluation suite

## Commands

```bash
make up                                       # build + start the container once
make evaluation-nlseeker [CONFIG=path.yaml]   # fresh run; default CONFIG=evaluation/nlseeker/configs/default.yaml
make evaluation-nlseeker RESUME=<run-dir>     # resume a cancelled/crashed run (dir name under results/, .partial included)
make eval-clean RUN=<run-dir-name>            # rm results/<RUN>, routed through the container
make eval-clean-cache                         # rm the extracted lake/bench cache (datasets/ is never touched)
```

`RESUME` skips datasets already marked `complete` in the run's `run_meta.json`,
wipes and fully rebuilds the one that was interrupted (index included - no
partial reuse), and takes its configuration from the `config.yaml` copied into
the run dir, so `CONFIG` is ignored on resume.

## Configs

| config | grid | maps to |
| --- | --- | --- |
| `configs/default.yaml` | k [1, 5], n [5], alpha [0.5], all five datasets | Pneuma paper Figs 6-8 (main results) |
| `configs/alpha_sweep.yaml` | k [1], n [5], alpha [0.0 .. 1.0] | Pneuma paper Fig 14 |
| `configs/n_sweep.yaml` | k [1], n [1, 5, 10, 15, 20], alpha [0.5] | Pneuma paper Fig 16 |
| `configs/smoke.yaml` | adventure_works only, BC1+BX1, questions_limit 2 | quick end-to-end check, not a paper figure |

## Results layout

```
data/Evaluation/NLSeeker/results/<ts>__<tag>/
  config.yaml              # byte copy of the input YAML
  run_meta.json            # input sha256s, endpoint/package versions, per-dataset status
  summary.csv              # one row per dataset x family x k x n x alpha
  summary.md
  <dataset>/
    artifacts/
      eval.duckdb
      nl_index/
      nlseeker.ini
    per_query.jsonl        # qid, family, k, n, alpha, retrieved/answer ids, hit, rr, latency_ms
```

A run directory is created with a `.partial` suffix and renamed to drop it only
on completion, so a plain listing of `results/` shows which runs actually
finished. `data/Evaluation/NLSeeker/datasets/` is the source archive set and is
immutable - nothing in this suite writes to it or deletes from it.
