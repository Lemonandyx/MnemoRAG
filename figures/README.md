# Plotting programs

This directory contains plotting code only. It does not contain measurements,
request traces, aggregate result tables, model outputs, or dataset samples.

Each program reads an external CSV or a directory of result summaries and
writes PNG and PDF files below an ignored output directory. Dataset names,
series labels, panel counts, and axis ranges are derived from the supplied
input wherever possible.

```bash
python figures/end_to_end/plot.py --input /path/to/table.csv
python figures/component_ablation/plot.py --input /path/to/table.csv
python figures/motivation/plot_hotness.py --source-csv /path/to/table.csv \
  --output-base output/hotness
python figures/retrieval_cache/plot.py /path/to/summaries \
  --output output/retrieval
```

The repository intentionally excludes the inputs used to render manuscript
figures.
