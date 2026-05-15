# Q-Surv

This repository contains the code used to run the Q-Surv survival modeling benchmark. It covers both tabular and medical imaging datasets, and keeps the model comparison on a shared path for data loading, training, hyperparameter search, and evaluation.

The main benchmark compares Q-Surv variants with Cox-style and neural survival baselines. The code is organized so that new runs write a compact JSON result per dataset, model, and seed, which can then be summarized into the final result tables.

## What Is Here

- `main.py` is the primary benchmark entry point for training and evaluation.
- `simulation.py` runs the synthetic survival simulations.
- `summarize.py` turns completed benchmark JSON files into `output/summary_all_horizons.csv`.
- `data/` contains dataset loaders and preprocessing scripts.
- `models/` contains survival model wrappers.
- `networks/` contains neural backbones and time-conditioning modules.
- `tools/eval.py` contains concordance, Brier/log-loss, D-calibration, and clustering utilities.

## Installation

Start from a clean Python environment. Install PyTorch in the way that matches your hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

Data files are not included in the repository. Keep downloaded or generated files under `data/files/`, `data/*/images/`, or the relevant dataset subdirectory. These paths are ignored by git so the repo stays lightweight.

For the public tabular datasets that can be fetched from Python packages or direct download, run:

```bash
python data/prepare.py
```

That command writes Metabric, SUPPORT2, FLChain, and GBSG2 CSVs to `data/files/`. NWTCO and Framingham require separate handling:

- NWTCO is included in the CRAN/R `survival` package as `nwtco`. Export it to CSV with:

  ```bash
  Rscript -e 'data(nwtco, package="survival"); write.csv(nwtco, "data/files/nwtco.csv", row.names=FALSE)'
  ```

- Framingham comes from the Framingham Heart Study longitudinal teaching dataset available through NHLBI BioLINCC's teaching datasets page. Request the teaching dataset from BioLINCC, then save or rename the downloaded CSV to `data/files/framingham.csv`.

After setup, the two manually supplied files should be:

```text
data/files/nwtco.csv
data/files/framingham.csv
```

Active tabular datasets are `metabric`, `support2`, `flchain`, `gbsg`, `nwtco`, and `framingham`.

Active imaging datasets are `covid`, `c4kc_kits`, and `brats`. Their loaders expect local clinical files and preprocessed images. The relevant preprocessing entry points are:

```bash
python data/c4kc_kits/download_and_preprocess.py
python data/brats/preprocess.py
python data/covid/process_clinical.py
python data/covid/process_ct.py
```

## Running Benchmarks

The commands below assume you are running from the repository root.

To run one dataset, model, and seed:

```bash
python main.py --dataset metabric --model qsurv --seed 0 --device cpu
```

To run several models or seeds at once:

```bash
python main.py --dataset framingham --model coxtime qsurv qsurv_film --seed 0 1 2
```

To run an imaging benchmark:

```bash
python main.py --dataset brats --model qsurv deephit --seed 0 --device cuda --pretrained
```

Available model names are `coxcc`, `coxtime`, `nnetsurv`, `mdn`, `soden`, `desurv`, `qsurv`, `qsurv_concat`, `qsurv_film`, and `deephit`. Passing `--model all` runs the default benchmark model set.

Results are written as:

```text
output/{dataset}+{model}+{seed:02d}_hpo.json
```

Existing result files are skipped by default. Use `--force` to rerun completed cells.

## Summarizing

Once benchmark runs finish, aggregate the result JSON files with:

```bash
python summarize.py
```

The summary script reads `output/*_hpo.json`, prints the per-horizon metric tables, and writes:

```text
output/summary_all_horizons.csv
```
