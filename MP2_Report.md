# Selecting Reliable LLM Reasoning

DASE7506 Mini Project 2 | Student ID 3036805250 | 29 October 2026

## Abstract

I study whether fixed solution embeddings and answer agreement can select reliable mathematical answers under a small CPU inference budget. A logistic regression selector trained on questions with competing parsed answers, using original and within-question centered embeddings, achieves 742/798 (92.98%) validation accuracy and 377/467 (80.73%) official test accuracy. Full local prediction plus scoring takes 1.165 seconds, and the required inference assets occupy 0.03975 MiB uncompressed. Larger margins between distinct answer scores correspond to higher validation accuracy, but high-confidence errors and a parser failure show that confidence is not a guarantee of valid reasoning.

## Task and research questions

Each question has four candidate solutions and four fixed 1024-dimensional embeddings. The selector returns one index from 0 to 3. Multiple candidates may earn credit; correctness follows the supplied numeric parser and reference answer, rather than a judgment of every reasoning step. No new language-model generation, external data, or additional pretrained weights are used.

The experiments ask whether embeddings improve over answer voting, whether training on questions with competing answers helps, whether centered features improve selection, and whether answer-score margins identify unreliable choices. A listwise ranking objective is also examined as a negative result. Later experiments compare answer pooling, pairwise ranking and a small neural classifier.

| Split | Questions | With competing parsed answers |
| --- | --- | --- |
| Training | 4,755 | 2,114 |
| Validation | 798 | 355 |
| Test | 467 | All, by the supplied guide |

A question has competing answers when at least two distinct non-null answers are returned by common.parse_answer. There are 2,641 training and 443 validation questions with just one distinct valid answer. Under the supplied guarantee that every group contains a correct candidate, choosing any parseable candidate in these one-answer groups earns credit. This makes the disagreement subset a more informative development diagnostic. All 798 validation questions remain in the official score.

# Selection method and training

## Labels and preprocessing

For each candidate, I compare its official parsed answer with the parsed training reference. A candidate is positive only when the answer is non-null and equal to the reference. All four candidates of each selected training question are retained, including unparseable negative examples. Filtering questions uses candidate answers only. StandardScaler is fitted exclusively on the selected training candidates.

## Features and scoring

For a candidate embedding, I concatenate the original 1024 coordinates with its difference from the mean embedding of the four candidates in that question. This yields 2048 features. The final classifier is scikit-learn LogisticRegression with L2 regularization, the lbfgs solver, max_iter=2000 and random_state=17. The selected C is 0.001.

Inference standardizes these features using saved training statistics and computes the classifier decision score. An agreement bonus of 0.25 is added for each other candidate with the same parsed answer. Unparseable candidates receive negative infinity. The highest-scoring candidate is selected; ties favor the first candidate. If none is parseable, index 0 is returned, although such a group cannot earn credit under the supplied scorer.

## Interpretation of centered features

The final scorer remains linear. Its centered-feature contribution consists of a candidate-dependent linear term and a question-level term shared by all four candidates. The shared term cancels when candidates are ranked. Thus, the observed gain does not establish nonlinear cross-candidate reasoning. Duplicating and transforming the representation changes the fitted classifier and effective regularization; it can also affect the learned question-level offset during binary training. These are plausible explanations, not separately established mechanisms.

## Model selection

For each logistic model, C is searched over 0.001, 0.01, 0.1 and 1.0. The initial embedding-only model uses full validation accuracy. For the disagreement-trained models, agreement weights are searched over 0, 0.25, 0.5, 1, 2 and 4. Selection maximizes disagreement validation correct count, then full validation correct count; remaining ties keep the earlier setting. The final model uses 2,114 training questions, or 8,456 candidate examples. There is no retraining on validation references.

Only numeric weights, a bias, standardization statistics and a configuration are required for inference. relative_selector.py imports agreement and selection helpers from fusion_selector.py and the fixed parser from common.py. It accepts the bank and arrays passed by the evaluator and does not read reference answers or use IDs for prediction.

# Experimental results and ablations

| Method | All validation | Disagreement only |
| --- | --- | --- |
| Majority voting | 688/798  86.22% | 245/355  69.01% |
| Embedding logistic regression | 712/798  89.22% | 270/355  76.06% |
| Initial fusion | 724/798  90.73% | 281/355  79.15% |
| Disagreement training and fusion | 732/798  91.73% | 289/355  81.41% |
| Centered feature model | 742/798  92.98% | 299/355  84.23% |
| Listwise ranker | 731/798  91.60% | 288/355  81.13% |

## Evidence for individual choices

The embedding-only model improves over voting by 24 validation questions. Their errors are complementary: both methods are correct on 646 questions; only voting is correct on 42, only logistic regression on 66, and neither on 44. Different selected indices alone are not evidence of different correctness because several candidates can share an answer.

For the original classifier, masking unparseable candidates improves the correct count from 712 to 715. Adding agreement with alpha=0.5 improves it to 724. Larger agreement weights hurt: alpha=4 yields 707 correct, consistent with correlated mistakes among candidate solutions.

At fixed C=0.01 and alpha=0.5, changing from all training groups to disagreement training groups increases disagreement correctness from 281 to 286. This supports a training-subset effect without attributing all gains from retuning to the subset. At fixed C=0.001 and alpha=0.25, adding centered features increases disagreement correctness from 288 to 299. Selected-model comparisons also reflect parameter search and are not pure single-factor ablations.

## Listwise ranking experiment

A separate linear ranker minimizes the negative log of the softmax probability mass assigned to the set of correct candidates, plus L2 regularization. Unparseable candidates are excluded from its denominator. A single seed-17 split of disagreement training questions gives 1,691 fitting and 423 development questions. Regularization values 0.1, 0.01 and 0.001 and agreement weights 0, 0.25, 0.5 and 1 are compared internally. The selected setting is 0.001 and 1.0; the model is then refitted on all 2,114 disagreement training questions.

All optimizer runs report convergence. Internal selection reaches 328/423 (77.54%); official validation reaches 731/798. This experiment does not improve on the centered-feature model. It is a single holdout experiment, not cross-validation, and the different selection protocols prevent a clean conclusion that the ranking objective itself is inferior.

# Additional experiments with training cross validation

## Protocol and scope

The 2,114 disagreement training questions are split into three shuffled folds with seed 17. Development fold sizes are 705, 705 and 704. Each question and all four candidates stay together; standardization is fitted within each fitting fold. Every question receives one out-of-fold prediction per setting. These training questions were used in earlier development, and the final baseline configuration was already chosen on official validation. This is a development comparison, not nested cross-validation or a new independent holdout.

| Method | Training OOF | Validation | Adopted |
| --- | --- | --- | --- |
| Existing baseline | 1699/2114 / 80.37% | 742/798 / 92.98% | Yes |
| Probability sum | 1656/2114 / 78.33% | Not evaluated | No |
| Probability mean | 1617/2114 / 76.49% | Not evaluated | No |
| Pairwise ranking | 1704/2114 / 80.61% | 740/798 / 92.73% | No |
| Small MLP | 1665/2114 / 78.76% | 728/798 / 91.23% | No |

The pooling search selects the existing baseline. Its refitted validation count is 742; sum and mean are not separately evaluated on official validation. Sigmoid scores are pooled without assuming candidate independence, and are not claimed to be calibrated probabilities.

## Designs and results

Pooling compares the baseline maximum score plus a 0.25 agreement bonus against sum and mean sigmoid scores within each answer group, with C fixed at 0.001. Sum and mean lose to the baseline in every fold. Pairwise training minimizes logistic loss on correct-minus-incorrect embedding differences, weights each question equally and excludes unparseable candidates from pairs. Three L2 strengths (0.1, 0.01, 0.001) and four agreement weights (0, 0.25, 0.5, 1) are compared. The selected strength 0.01 and weight 0.25 gain five OOF answers but lose two official validation answers; this is not convincing evidence for replacement.

The MLP uses original-plus-centered features, one ReLU layer of width 32 or 64, Adam, L2 alpha=1, batch size 128 and at most 150 epochs. Training-loss stopping is used; no candidate-level validation split is created. Agreement weights are 0, 0.25 and 0.5. Width 32 and weight 0.5 are selected. The selected-width fold runs stop at 73, 82 and 81 epochs, and NumPy export agrees with sklearn on the recorded check. Inferior development accuracy does not isolate its cause; overfitting remains a possible explanation.

Recorded full training/search durations are 3.88 seconds for pooling, 5.33 for pairwise ranking and 109.44 for the MLP, including refits. These are individual local runs. The reused folds and best-of-grid selection introduce optimism. No significance claim is made, and none of these additions is evaluated on test.

# Reliability and representative failures

## Margins between distinct answers

For each distinct parsed answer, I take the maximum final candidate score among candidates giving that answer. The confidence margin is the difference between the two highest distinct-answer scores. This avoids treating two candidates with the same final answer as competing conclusions. The 355 disagreement validation questions are sorted by this margin and split into four nearly equal groups.

| Margin group | Correct | Accuracy |
| --- | --- | --- |
| Lowest | 58/89 | 65.17% |
| Middle low | 74/89 | 83.15% |
| Middle high | 81/89 | 91.01% |
| Highest | 86/88 | 97.73% |

Accuracy increases across all four groups. The lowest group contains 31 of the 56 validation errors (55.36%), while the highest still contains two errors. Margins therefore provide a useful reliability ordering on this validation set, but are not calibrated probabilities. These data were also used in model development; independent confirmation is needed. The selector still returns an answer for every question and does not abstain.

## Shared misinterpretation

In the Lexie age problem, the brother is 8 minus 6, or 2, and the sister is twice 8, or 16; the difference is 14. Three candidates reverse the sister relationship and answer 2. The model selects one of them with a distinct-answer margin of 3.510. Agreement can reward a shared error; multiple matching traces are not independent evidence.

## Correct algebra for an incorrect model

Andy walks 50 meters to school, 50 back home, then to the market, for 140 meters total. The market distance is 40. The selected candidate instead solves 50 plus two times the market distance equals 140, obtaining 45. Its algebra is consistent, but its variable assignment is not. A checker of arithmetic alone would miss this error.

## Parser induced scoring failure

For the plumber problem, the selected candidate correctly obtains job earnings 240, 250 and 220 and ends with a boxed 250. However, it includes a Markdown heading with four hash marks followed by Job 3. The fixed parser prioritizes the last four-hash delimiter and extracts the first subsequent number, 3, rather than the final boxed answer. This selection is mathematically correct but receives no official credit. The parser and reported scores are kept unchanged. This is an extraction failure, not evidence that the selected reasoning is mathematically wrong.

# Evaluation resources and reproducibility

## Official test result

The selected centered-feature checkpoint obtains 377 correct choices out of 467, or 80.72805139186295%. An earlier fusion checkpoint obtained 360/467 (77.08779443254818%). The difference is 17 questions, or 3.64 percentage points. Neither number is presented as evidence from a permanently untouched test set; the development chronology is disclosed on the next page.

| Measurement | Recorded value |
| --- | --- |
| Local full prediction plus scoring | 1.1651251 seconds |
| Prediction internal timer | 0.1831439 seconds |
| Scoring internal timer | 0.0696240 seconds |
| Required assets uncompressed | 41,681 bytes  /  0.0397501 MiB |
| Local operating system | Windows 11 10.0.26200 |
| Python | 3.12.15  Anaconda |
| CPU identifier | AMD64 Family 25 Model 97 Stepping 2 |
| Logical processors and configured threads | 32 logical processors  /  4 threads |

The external timer includes fresh prediction and scoring process startup, imports, data and asset loading, preprocessing and execution. The measurement is a single local run, not a median or a reference-machine certification. The course Xeon reference machine determines compliance with the 30-second limit. The 16-MiB asset budget excludes supplied data and ordinary source. My model uses no ensemble or additional inference assets.

## Training and search costs

The final search uses four logistic fits and six agreement weights per fit, with optimizer iterations 40, 97, 212 and 382. A later four-thread rerun takes 10.4476 seconds; regenerated model, configuration and experiment-record hashes match the originals. This measures a rerun, not the unrecorded historical duration. The ranker separately records its training/search time in runs/ranker/training_results.json.

## Reproduction procedure

Use the original course data for protocol dase7506-mp2-open-hard-verifier-v5 and run check_data.py. From code/, install pinned dependencies and set thread limits to four. Run predict.py with --method relative_selector.py --checkpoint runs/relative, then score_predictions.py. The recorded clean-environment test reproduction gives the same 467 choices and 377 correct answers. Evaluation requires NumPy; training also uses scikit-learn and SciPy. The root README gives exact commands and all experimental entry points.

The checkpoint comprises runs/relative/model.npz and config.json. ASSET_MANIFEST.json records full SHA256 hashes. The model hash begins b38edb7987ef7f75; the configuration hash begins 4d33c0cbebfcb84e. The submitted repository must retain these matching assets and identify its immutable commit. A peer can reproduce checkpoint evaluation without retraining.

# Limitations disclosure and conclusions

## Development chronology

An initial fusion method was tested before development was complete, scoring 360/467. Seeing that score motivated further development. Disagreement training and centered features were then fitted and selected using training and validation data; the selected method scored 377/467 on test. A later listwise ranker was not adopted. After the second test, pooling, pairwise ranking and MLP experiments used training cross-validation and official validation only. None replaced the centered-feature model or received a test evaluation. A fresh local environment reproduced the unchanged final checkpoint result.

The first public test result nevertheless influenced the decision to continue development. Thus, the public test bank is not an untouched independent holdout for the whole research process. Repeated validation-based selection can also introduce optimism, particularly with only 355 disagreement validation questions. All recorded evaluations are retained. A new audit bank is needed for a stronger generalization assessment; no claim of complete test independence is made.

## Scope of the findings

The test bank is restricted to competing parsed answers, whereas more than half of validation questions have one valid answer. This composition difference helps explain why the full validation score is a poor standalone expectation for test performance, but does not quantify the entire gap. Reported gains come from one seed and this course bank; no statistical significance or broad reasoning competence is established.

I have not isolated the effects of candidate length, formatting or confident wording. Whole-trace embeddings may capture these shortcuts. The classifier does not explicitly verify symbolic derivations, and centered features preserve a linear ranking form. Official numeric correctness also differs from human judgments when reasoning is flawed or parsing fails. These limitations constrain the interpretation of both accuracy and confidence.

## AI assistance and reused materials

ChatGPT provided substantial assistance with method suggestions, code drafting, debugging guidance, experiment interpretation and report preparation. The recorded executions were performed locally by the student; this assistance does not constitute independent verification of every result. The student is responsible for reviewing and understanding the implementation and checking this report before submission. Supplied code, parser, scoring tools, embeddings and data are reused from the course repository. Their license and attribution files must be retained. No external training data or pretrained weights were added.

## Conclusion

The strongest observed development result combines training on competing-answer groups, a transformed embedding representation and a modest agreement bonus. It attains 80.73% official test accuracy with very small inference assets and low local CPU cost. Reliability analysis is more informative than accuracy alone: distinct-answer margins separate easier from riskier choices, while shared misinterpretations and parser behavior explain why high confidence can still fail.

## Sources

Course guide: github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md; running instructions: code/README.md in that repository. Lecture 6, From Comparisons to a Scalar Reward, motivates the pairwise extension. Questions derive from GSM8K; supplied candidates and embeddings use Qwen3-0.6B. Course license and attribution files are retained. Numerical evidence is in code/runs/ and ASSET_MANIFEST.json.
