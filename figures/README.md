# Plotting programs

Plotting programs for the MnemoRAG paper. Each program reads a CSV or a
directory of result summaries and exports PNG and PDF files.

```bash
python figures/end_to_end/plot.py --input /path/to/table.csv
python figures/component_ablation/plot.py --input /path/to/table.csv
python figures/motivation/plot_hotness.py --source-csv /path/to/table.csv \
  --output-base output/hotness
python figures/retrieval_cache/plot.py /path/to/summaries \
  --dataset dataset_name --output output/retrieval
```
