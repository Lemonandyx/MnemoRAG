# Paper plotting programs

Plotting code is grouped by the experimental claim it supports. Experimental
tables, raw request traces, and model outputs remain outside the repository.

| Directory | Input | Output |
|---|---|---|
| `end_to_end/` | End-to-end aggregate CSV | Dataset/workflow latency grid |
| `component_ablation/` | Component aggregate CSV | Latency speedup and quality panels |
| `motivation/` | Background aggregates or raw campaign summaries | Similarity, hotness, stage latency, and recall figures |
| `retrieval_cache/` | Retrieval campaign summaries | nprobe/request-rate comparison |

Paper-level aggregate plots accept their input tables explicitly:

```bash
python figures/end_to_end/plot.py --input /path/to/end_to_end.csv
python figures/component_ablation/plot.py --input /path/to/component_ablation.csv
```

All programs write PNG and PDF outputs under `output/`, which is ignored by
Git. Figure scripts that consume raw campaign summaries expose `--help` with
their expected schema.
