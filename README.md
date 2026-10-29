# DASE7506 MP2 Selecting Reliable LLM Reasoning

**Student ID:** 3036805250  
**Protocol:** `dase7506-mp2-open-hard-verifier-v5`  
**Official test accuracy:** **80.72805139186295% (377/467)**

This repository contains the candidate-selection implementation, matching
checkpoints, supplied course data, experimental records and report. The final
method selects one of four supplied candidate solutions without generating new
text or consulting reference answers during prediction.

## Final submission artifacts

| Item | Location |
| --- | --- |
| Report, 7 pages | [MP2_Report.pdf](MP2_Report.pdf) |
| Final prediction implementation | [code/relative_selector.py](code/relative_selector.py) |
| Final checkpoint | [code/runs/relative/model.npz](code/runs/relative/model.npz) |
| Final configuration | [code/runs/relative/config.json](code/runs/relative/config.json) |
| Official test scoring JSON | [code/runs/relative-test-score.json](code/runs/relative-test-score.json) |
| Test predictions | [code/runs/relative-test-predictions.json](code/runs/relative-test-predictions.json) |
| Fresh-environment reproduction | [code/runs/clean-test-score.json](code/runs/clean-test-score.json) |
| Full inference resource record | [code/runs/relative-final-evaluation-record.json](code/runs/relative-final-evaluation-record.json) |
| Supplementary training timing | [code/runs/supplementary-retraining-resources.json](code/runs/supplementary-retraining-resources.json) |
| Inference asset hashes | [code/ASSET_MANIFEST.json](code/ASSET_MANIFEST.json) |
| Python and dependency file hashes | [code/SOURCE_SHA256.json](code/SOURCE_SHA256.json) |
| Experiment summary | [code/EXPERIMENT_SUMMARY.json](code/EXPERIMENT_SUMMARY.json) |
| AI assistance and reuse disclosure | [code/AI_ASSISTANCE.md](code/AI_ASSISTANCE.md) |
| Project guide | [guide/GUIDE.md](guide/GUIDE.md) |
| Official running instructions | [code/README.md](code/README.md) |

Use **relative_selector.py with runs/relative/** for the final result.
`code/submission.py` is the unchanged empty course interface, so it must not be
used as the method entry point. Other checkpoints are retained for experimental
reproduction and are not loaded by the final selector.

## 1. Environment and installation

The recorded training environment is Windows 11, Anaconda Python 3.12.15,
NumPy 2.5.3, scikit-learn 1.9.1 and SciPy 1.18.1. The direct dependencies are
pinned in `code/requirements-method.txt`; `code/requirements-lock.txt` preserves
the full earlier environment snapshot.

From the repository root, in **Anaconda Prompt**:

```bat
conda create -n mp2-reproduce python=3.12 -y
conda activate mp2-reproduce
cd code
python -m pip install -r requirements-method.txt
python check_data.py
```

Expected data check: `status: passed`, with 4,755 training, 798 validation and
467 test questions, and embeddings shaped `[N, 4, 1024]`.

Set thread limits before training or evaluation:

```bat
set OMP_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set MKL_NUM_THREADS=4
set NUMEXPR_NUM_THREADS=4
set BLIS_NUM_THREADS=4
set VECLIB_MAXIMUM_THREADS=4
```

For **Linux/macOS**, from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
cd code
python -m pip install -r requirements-method.txt
python check_data.py
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
```

The Python commands in the following sections are identical on both platforms.
The exact pinned environment was reproduced locally on Windows. Checkpoint
prediction and scoring were also independently reproduced in a Linux review
environment with Python 3.12.14 and NumPy 2.3.5. This does not establish that a
fresh POSIX installation of every pinned training dependency has been verified.

Final checkpoint inference needs NumPy and the standard library only.
scikit-learn is used for training; SciPy is additionally used by ranking
experiments. Prediction and scoring work offline after installation.

## 2. Reproduce the final result without retraining

Run from `code/`, with the thread settings above:

```bat
python predict.py --method relative_selector.py --checkpoint runs/relative --split validation --output runs/reproduced-validation-predictions.json
python score_predictions.py --split validation --predictions runs/reproduced-validation-predictions.json --output runs/reproduced-validation-score.json
python predict.py --method relative_selector.py --checkpoint runs/relative --split test --output runs/reproduced-test-predictions.json
python score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json
```

| Split | Correct | Official accuracy |
| --- | ---: | ---: |
| Validation | 742/798 | 92.98245614035088% |
| Test | 377/467 | 80.72805139186295% |

The `accuracy` field in the official scoring JSON is the submission score.
The saved fresh-environment test predictions match all 467 original choices.
A subsequent independent checkpoint evaluation reproduced the same validation
and test choices and scores. Retraining is not necessary for reproduction.

For a new audit bank, use the same frozen method:

```bat
python predict.py --method relative_selector.py --checkpoint runs/relative --split audit --data-dir PATH_TO_AUDIT_INPUTS --output runs/audit-predictions.json
```

Replace `PATH_TO_AUDIT_INPUTS` with the supplied audit input directory.
The selector uses the bank and arrays passed to it; it does not assume public
question IDs or a fixed bank size.

Optional saved-record and file-identity check:

```bat
python verify_submission.py
```

This checks source/asset hashes, saved prediction structure and equality of the
original and clean-environment test choices. It is not a substitute for the
official prediction and scoring commands.

## 3. Method

Training labels compare each candidate's parsed numeric answer with the parsed
training reference using the unchanged `common.parse_answer`. Multiple correct
candidates are allowed. Unparseable candidates are negative examples.

The final model uses the 2,114 training questions that contain at least two
distinct non-null parsed candidate answers. All four candidates in each selected
question are retained, giving 8,456 candidate examples. Filtering questions uses
candidate answers only.

Each 1,024-dimensional supplied embedding is concatenated with its difference
from the mean of the four embeddings for that question. The resulting 2,048
features are standardized using statistics fitted on training examples only.
A logistic regression classifier uses `solver="lbfgs"`, `C=0.001`,
`max_iter=2000` and `random_state=17`.

At inference, the classifier decision score receives a bonus of **0.25 for each
other candidate with the same parsed answer**. Unparseable candidates are
masked. The highest-scoring candidate is returned; exact ties favor the first
candidate. If every candidate is unparseable, the fallback is index 0.

The scorer remains linear. The common question-level mean contribution cancels
in within-question ranking, so gains from this representation do not establish
nonlinear cross-candidate reasoning. The report discusses training and
regularization effects as possible explanations.

No external training data, additional pretrained weights, online model calls,
answer lookups or cached predictions are used by the final method. IDs in
records are used for alignment and reproducibility, not prediction.

## 4. Training and experiment reproduction

Use a separate copy of the repository for retraining: scripts can overwrite
checkpoints and historical records under `runs/`. Keep the submitted checkpoint
and its matching hashes unchanged in the copy used for checkpoint evaluation.

To reproduce the final model's training and selection, from `code/`:

```bat
python train_relative.py
```

The search uses `C` in `{0.001, 0.01, 0.1, 1.0}` and agreement weight in
`{0, 0.25, 0.5, 1, 2, 4}`. It maximizes disagreement-validation correct count,
then full-validation correct count; remaining ties keep the earlier setting.
The selected configuration is `C=0.001`, agreement weight `0.25`.
Validation references select settings but do not fit weights or standardization.

To reproduce the full sequence of experiments and analyses:

```bat
python train_linear.py
python tune_fusion.py
python train_hard_linear.py
python train_relative.py
python train_ranker.py
python analyze_difficulty.py
python analyze_relative.py
python train_group_pool.py
python train_pairwise.py
python train_mlp.py
```

### Initial experiments

| Method | Validation correct | Accuracy | Prediction method | Checkpoint |
| --- | ---: | ---: | --- | --- |
| Majority vote | 688/798 | 86.22% | baseline_vote.py | None |
| Original linear classifier | 712/798 | 89.22% | linear_selector.py | runs/linear |
| Original fusion | 724/798 | 90.73% | fusion_selector.py | runs/fusion |
| Disagreement-trained linear fusion | 732/798 | 91.73% | fusion_selector.py | runs/hard_linear |
| Final centered-feature classifier | 742/798 | 92.98% | relative_selector.py | runs/relative |
| Listwise ranker | 731/798 | 91.60% | fusion_selector.py | runs/ranker |

The listwise ranker uses a single seed-17 internal split of 1,691 fitting and
423 development questions, not cross-validation. Its regularization grid is
`{0.1,0.01,0.001}` and agreement grid is `{0,0.25,0.5,1}`. The selected setting
is `0.001` and `1.0`, followed by a refit on all disagreement training groups.
It minimizes the negative log probability mass of the correct candidate set,
with L2 regularization. It is not the later pairwise model.

### Additional experiments

Pooling, pairwise ranking and MLP use the same three question-level folds of
the 2,114 disagreement training questions: `KFold(n_splits=3, shuffle=True,
random_state=17)`. Development folds contain 705, 705 and 704 questions.
Each question stays within one fold with all four candidates. Scaling is fitted
only on each fitting fold. Selected settings are refitted on all disagreement
training questions, then evaluated on official validation.

| Method | Training out-of-fold correct | Official validation correct |
| --- | ---: | ---: |
| Existing baseline | 1699/2114 | 742/798 |
| Sigmoid probability sum | 1656/2114 | Not separately evaluated |
| Sigmoid probability mean | 1617/2114 | Not separately evaluated |
| Pairwise linear ranking | 1704/2114 | 740/798 |
| Small MLP | 1665/2114 | 728/798 |

- **Pooling:** holds `C=0.001` fixed and compares the baseline with sum and mean
  sigmoid scores within answer groups. The scores are not claimed to be
  calibrated probabilities. Baseline is selected; sum/mean are not separately
  evaluated on official validation. The experimental entry point is
  `group_selector.py` with `runs/group_pool`.
- **Pairwise ranking:** uses original 1,024-dimensional embeddings, logistic
  loss on valid correct-minus-incorrect pairs and equal total weight per
  question. L2 strengths are `{0.1,0.01,0.001}` and agreement weights are
  `{0,0.25,0.5,1}`. The selected values are `0.01` and `0.25`. Use
  `fusion_selector.py` with `runs/pairwise`.
- **MLP:** uses original-plus-centered features, one ReLU hidden layer of width
  `{32,64}`, Adam, weight regularization `1.0`, batch size `128`, learning rate
  `0.001`, maximum `150` epochs and seed `17`. Stopping monitors training loss
  (`tol=0.0001`, patience `15`); sklearn `early_stopping=False` avoids an
  automatic candidate-level development split. Agreement weights are
  `{0,0.25,0.5}`; width `32` and weight `0.5` are selected. NumPy export passed
  the recorded forward comparison. Use `mlp_selector.py` with `runs/mlp`.

For an experimental checkpoint, use the official predict/score commands in
Section 2 with its method, checkpoint and distinct output filenames, using
`--split validation`. For majority voting, omit `--checkpoint`.

These are development comparisons with reused data and selected grids, not
nested cross-validation or evidence of statistically significant improvements.
None of these added methods replaces the final relative checkpoint, and none
was evaluated on the public test bank.

## 5. Resources and hashes

The two required final inference assets are:

| Asset | Uncompressed bytes | SHA256 |
| --- | ---: | --- |
| runs/relative/model.npz | 41,476 | b38edb7987ef7f75d724dfe7111031b3e9aa9675a732b31e0877d6fc66014f1b |
| runs/relative/config.json | 205 | 4d33c0cbebfcb84eb57c6e13f4808a976e5807e386b97e2ddd7f2636b2d7a972 |

Total: **41,681 uncompressed bytes = 0.039750099 MiB**. NPZ member sizes are
counted uncompressed. Supplied data and ordinary source code are excluded.
Alternative experimental checkpoints are not inference assets of the final
method because it does not load them. No ensemble is used.

The recorded full fresh-process prediction plus independent scoring time is
**1.1651251 seconds**, including process startup, imports, data/asset loading,
preprocessing and scoring. Local hardware: Windows 11, AMD64 Family 25 Model 97
Stepping 2, 32 logical processors, thread limits set to four. This is a single
local observation, not reference-machine certification. The course reference
machine determines compliance with the 30-second limit.

The final training search uses four classifier fits and six agreement settings
per fit. A later four-thread rerun took **10.4475709 seconds**, and regenerated
model/configuration/experiment-record hashes matched the originals. This is a
later rerun measurement; historical full logistic development times were not
recorded. The supplementary record preserves that distinction.

| Additional experiment | Recorded full training/search time | Fits including refit |
| --- | ---: | ---: |
| Pooling | 3.8796 seconds | 4 |
| Pairwise | 5.3316 seconds | 10 |
| MLP | 109.4417 seconds | 7 |

Training scripts use the caller's thread environment settings. Additional timing
records do not independently measure effective BLAS thread counts. Search
settings, optimizer iterations, seeds and recorded durations are retained in
`runs/`. Hardware and software differences may change timings, iterations and
retrained numerical weights. Use the supplied checkpoint for direct scoring
reproduction.

`evaluate_relative.py` measures complete test prediction/scoring but overwrites
its corresponding original timing records. Archive those records first if
rerunning it. `measure_retraining.py` backs up and restores the final checkpoint;
archive the existing supplementary timing JSON before attempting another run.
Neither helper is needed for ordinary checkpoint evaluation.

## 6. Data provenance and fixed evaluation

The supplied data and evaluation files are retained unchanged. Data and feature
integrity are checked by `python check_data.py`. Upstream file hashes are listed
in `PACKAGE_MANIFEST.json` (in the package root or its original code/ location).
The report contains ablations, reliability analysis, failure cases and the
limitations of numeric-answer scoring.

Questions and references derive from GSM8K. Candidate solutions and fixed
embeddings were prepared with Qwen3-0.6B by the course authors. Retain
[code/data/LICENSE.txt](code/data/LICENSE.txt) and
[code/data/NOTICE.txt](code/data/NOTICE.txt), together with the original
manifests and attribution files.

The official parser/scorer is not modified. In one validation case, a selected
solution ends with the mathematically correct boxed answer 250 but is parsed as
3 because of a four-hash Markdown heading. It remains incorrect under official
scoring. No reference-based correction is applied during prediction.

Course materials:
- [Original project repository](https://github.com/xudongwu-0/DASE7506-MP2)
- [Project guide](guide/GUIDE.md)
- [Official installation and evaluation instructions](code/README.md)

## 7. Development chronology and AI assistance

An early fusion method was tested publicly and scored 360/467. That score
motivated further development. The relative model was subsequently selected
using validation results and then tested, scoring 377/467. Later listwise,
pooling, pairwise and MLP experiments were not adopted and were not evaluated
on the public test bank. A fresh local environment reproduced the unchanged
final checkpoint rather than testing a new candidate.

The public test bank is therefore not an untouched holdout for the entire
research process. Both test records are retained. No test reference is used
in weight fitting or parameter-selection computations; the report nevertheless
discloses that the earlier test score influenced the decision to continue.
Repeated development selection also limits generalization claims.

ChatGPT provided substantive assistance with method suggestions, code drafting,
debugging guidance, result interpretation and report preparation. The student
executed the recorded experiments and is responsible for understanding and
validating the implementation and report. See
[AI_ASSISTANCE.md](code/AI_ASSISTANCE.md) for the full disclosure.

## 8. Submitted version and reproduction links

The submission should identify one immutable Git commit and its matching final
checkpoint. From the repository root, obtain the exact version with:

```bash
git rev-parse HEAD
```

Use the public URL of that commit as the code link and the final checkpoint
folder at that same commit as the asset link. Do not substitute experimental
checkpoints. The full commit identifier and links belong in the submission
entry; a changing branch name alone does not identify a frozen version.

Retain the report, data/licenses, training and inference source, pinned
dependencies, official evaluation JSON, experiment/resource records and hashes
in the public repository. Check that links work without signing in. Finally,
create the GitHub issue prepared by the course submission website; preparing
its form or link alone does not complete submission.
