# MantisV2 controlled adaptation experiment

This project runs the matched experiment

1. `frozen`: pretrained MantisV2, frozen encoder, train `LayerNorm + Linear`;
2. `full`: the same pretrained MantisV2, train the complete encoder and head;
3. `scratch`: the same MantisV2 architecture from random initialization, train
   the complete encoder and head.

The implementation calls the authors' `MantisV2` and `MantisTrainer` classes
directly. It pins the official code revision associated with `mantis-tsfm`
v1.0.0 and the MantisV2 checkpoint revision. See
[`docs/OFFICIAL_CODE_AUDIT.md`](docs/OFFICIAL_CODE_AUDIT.md) for the source
audit and the reasons the sktime wrapper is not used for the primary results.

## Reproduction contract

The three modes share the same:

- official MantisV2 architecture (`return_transf_layer=-1`,
  `output_token="cls_token"`);
- length-512 linear interpolation;
- complete six-block forward path;
- channel-wise shared encoder and concatenation for multivariate inputs;
- `LayerNorm + Linear` prediction head;
- cross-entropy loss;
- AdamW, learning rate `2e-4`, weight decay `0.05`, betas `(0.9, 0.999)`;
- 100 epochs, batch size 128, no early stopping, and last-epoch scoring;
- official 10-epoch warm-up plus cosine decay from `MantisTrainer`;
- train/test split and matched seeds `0,1,2`.

Only encoder initialization and the set of updated parameters differ. Test
labels are used only for final scoring.

The MantisV2 paper reports this fine-tuning protocol on UCR. Applying it to
multivariate tasks is an explicit benchmark extension; the code uses the
official Mantis package's multivariate interface rather than inventing a new
channel module.

## 1. Install

Python 3.10--3.12 and a CUDA-enabled PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch build appropriate for the machine first.
python -m pip install -e '.[archive,test]'
```

The project dependency pins official Mantis code commit
`0c94f8ceb9f1d1421dd292ed917090df8c31605b` (tag `v1.0.0`). The checkpoint is
loaded at revision `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6`.

Verify the environment and the true scratch/pretrained distinction:

```bash
python -m mantis_adaptation.verify_environment --device cpu
```

## 2. Prepare a dataset manifest

For the existing 198-task benchmark, the safest route is to point at its
already-fixed splits. Each local dataset directory must contain:

```text
DatasetName/
  X_train.npy    # (n, channels, time) or (n, time)
  y_train.npy
  X_test.npy
  y_test.npy
```

Generate a manifest by scanning the benchmark root:

```bash
python scripts/make_manifest.py \
  --data-root /path/to/benchmark \
  --output configs/datasets.csv
```

The scanner also accepts `.npz` datasets containing the four keys above. To
download by archive name through `aeon`, use rows with `source=aeon`; see
`configs/datasets.example.csv`. Local benchmark splits are preferred because
they lock the exact 198-task universe already used in the paper.

The loader rejects NaNs instead of silently adding an imputation rule. If the
existing benchmark uses imputation, export its post-preprocessing arrays so all
three modes receive exactly the same values.

For the 198-dataset UCR + UEA + MVCore66 benchmark stored under three server
paths, build the combined manifest directly:

```bash
python scripts/make_benchmark_manifest.py \
  --ucr-root /data/benchmarks/UCR \
  --uea-root /data/benchmarks/UEA \
  --mvcore66-root /data/benchmarks/MVCore66
```

This writes `configs/datasets.csv`, labels every row with its benchmark, checks
that exactly 198 datasets were found, and prefixes only cross-benchmark name
collisions. Add `--prefix-all` to prefix every dataset identifier, or
`--expected-count 0` to disable the count check.

## 3. Build and run the matrix

```bash
python scripts/build_tasks.py \
  --manifest configs/datasets.csv \
  --output configs/tasks.csv \
  --modes frozen full scratch \
  --seeds 0 1 2

# One run (task indices are zero-based)
python -m mantis_adaptation.run_one \
  --task-file configs/tasks.csv \
  --task-index 0 \
  --output-root results

# Sequential local execution with resume
bash scripts/run_matrix.sh configs/tasks.csv results

# Slurm array
bash scripts/submit_slurm.sh configs/tasks.csv results
```

Completed JSON files are skipped by default. Add `--overwrite` to rerun a
task. A failed task writes a failure record with its traceback, allowing the
matrix to continue and the missing cells to be audited.

### Memory-safe gradient accumulation

For high-channel datasets, keep the optimizer batch at 128 while reducing the
physical batch. The learning-rate schedule advances once per optimizer update,
not once per micro-batch:

```bash
CUDA_VISIBLE_DEVICES=7 python -m mantis_adaptation.run_one \
  --task-file configs/tasks.csv \
  --task-index 0 \
  --output-root results \
  --device cuda \
  --effective-batch-size 128 \
  --micro-batch-size 1 \
  --overwrite
```

The manifest/task CSV may instead contain optional `effective_batch_size` and
`micro_batch_size` columns. `scripts/build_tasks.py` preserves these columns.
The result JSON records both batch sizes and the accumulation step count.

For multiple local GPUs, partition `configs/tasks.csv` into one file per GPU,
or submit the supplied Slurm array. Do not change batch size only for a subset
of datasets if the primary claim is a strictly fixed protocol; report any
resource-driven deviation as a separate sensitivity run.

## 4. Aggregate results

```bash
python -m mantis_adaptation.summarize \
  --results-root results \
  --task-file configs/tasks.csv \
  --output-dir analysis \
  --bootstrap-repetitions 10000 \
  --bootstrap-seed 20260917
```

Outputs:

- `analysis/per_run.csv`: every seed-level run;
- `analysis/per_dataset.csv`: seed means and paired deltas;
- `analysis/main_table.csv`: mean accuracy, average rank, deltas, W/T/L, and
  gains/drops larger than 1 percentage point;
- `analysis/paired_effects.csv`: mean paired effects, 95% task-bootstrap CIs,
  W/T/L and negative-transfer rates;
- `analysis/grouped_effects.csv`: descriptive results by archive, training
  size, class count, and UV/MV;
- `analysis/delta_ecdf.pdf` and `.png`: distributions of `full-frozen` and
  `full-scratch`;
- `analysis/completeness.csv`: missing/failed run audit.

Bootstrap resampling uses datasets, not test samples, as the sampling unit.

## Directory layout

```text
mantis_adaptation/       experiment and analysis package
scripts/                 manifest, matrix, local and Slurm launchers
configs/                 example manifests and generated task matrices
docs/                    upstream-code audit and interpretation notes
tests/                   unit tests that do not require model downloads
```
