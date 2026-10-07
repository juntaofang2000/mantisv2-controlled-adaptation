# Official-code audit (2026-09-17)

## Sources checked

- Mantis authors' repository: `https://github.com/vfeofanov/mantis`, current
  main commit `9018b98b4c1e093d2fa618338695cd57146d3cd0`.
- Paper-release tag: `v1.0.0`, commit
  `0c94f8ceb9f1d1421dd292ed917090df8c31605b`.
- Official checkpoint: `paris-noah/MantisV2`, revision
  `8f6ca35cb54ab14b120618943c6fca5ddf5a76a6`.
- sktime repository: `https://github.com/sktime/sktime`, inspected at commit
  `6360ab5e65ed993730e54e8f6fff003caf193aeb`.
- MantisV2 paper, Appendix B.4: arXiv:2602.17868v1.

## What is consistent in sktime

`sktime.classification.foundation_models.mantis.MantisClassifier` imports
`MantisV2` and `MantisTrainer` from the authors' `mantis-tsfm` package. Its
length-512 interpolation also matches the official example (`linear`,
`align_corners=False`). The `full` and `head` strings are passed to the
official trainer with the expected broad meanings.

Therefore, sktime is a wrapper around the official backbone; it is not an
independent reimplementation of MantisV2.

## Why it is not paper-protocol equivalent by default

The inspected sktime defaults are 100 epochs and batch size 64, whereas
MantisV2 Appendix B.4 specifies 500 epochs and batch size 128.

More importantly, the wrapper does not expose the `head=` argument of
`MantisTrainer.fit`. This matters because:

- official `mantis-tsfm` v1.0.0 used `LayerNorm + Linear`, matching the paper;
- official v1.1.0 changed the default to `BatchNorm + Linear` and explicitly
  notes that this changes fine-tuning results;
- sktime declares `mantis-tsfm>=1.0.0`, so the head and results depend on which
  compatible package version happens to be installed.

The `scratch` documentation and behavior also diverge under the wrapper's
defaults. `_CachedMantisNetwork.load_network()` first loads `checkpoint` when
it is not `None`. `_build_trainer()` then constructs a fresh MantisV2 object
and copies those loaded weights for both `full` and `scratch`. The official
trainer treats the `scratch` string as a parameter-selection mode; it does not
reinitialize a pretrained network. Consequently,

```python
MantisClassifier(fine_tuning_type="scratch")
```

with the default `checkpoint="paris-noah/MantisV2"` starts from pretrained
weights, not random weights. `checkpoint=None` is required for a true scratch
run in this inspected implementation.

## Decision

Primary results use the authors' classes directly and pin the paper-release
revision. Every run explicitly supplies:

- `MantisV2(return_transf_layer=-1, output_token="cls_token")`;
- a `LayerNorm + Linear` head;
- AdamW with the paper hyperparameters;
- 100 epochs and batch size 128 for this controlled experiment;
- `learning_rate_adjusting=True`, which invokes the official v1.0.0 schedule;
- a pretrained checkpoint for `frozen`/`full`, and no checkpoint load for
  `scratch`.

This avoids both wrapper-default drift and the scratch-initialization trap.

## Gradient-accumulation extension

The official v1.0.0 `MantisTrainer.fit` performs one optimizer update per
physical DataLoader batch and does not expose gradient accumulation. This
project mirrors that pinned training loop locally when a smaller physical batch
is required. It preserves an effective batch of 128 by:

- accumulating summed cross-entropy gradients and normalizing by the actual
  number of accumulated samples, including the final partial batch;
- calling `optimizer.step()` once per effective batch;
- advancing the official 10-epoch warm-up and cosine schedule once per
  optimizer update rather than once per micro-batch;
- using the same physical batch for frozen, full and scratch runs selected for
  a memory-safe rerun.

With `micro_batch_size=effective_batch_size`, the local loop reduces to the
official per-batch update semantics. Both values and the derived accumulation
count are recorded in every result JSON.

## Scheduler clarification

Appendix B.4 says the authors “fix a fine-tuning scheme”; it does not say the
learning rate remains constant. At the v1.0.0 paper-release code, the
`MantisTrainer.fit` default is `learning_rate_adjusting=True`, implemented as a
10-epoch warm-up followed by cosine decay. The supplied experiment preserves
that official behavior. A constant-learning-rate study can be run separately,
but it should not be labeled the exact paper-release implementation.

## Scope clarification for multivariate tasks

The paper's reported fine-tuning result is on UCR. For a 198-task UV+MV study,
the multivariate part is a new benchmark extension. The official package
processes every channel with the same Mantis encoder and concatenates channel
embeddings before the head. This project uses that path unchanged and records
channel counts, but does not claim that the MantisV2 paper itself validated the
same fixed protocol on all 70 multivariate tasks.
