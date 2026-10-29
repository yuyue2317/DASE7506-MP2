# AI assistance and reuse disclosure

ChatGPT provided substantive help with experimental ideas, Python code drafting,
debugging guidance, result interpretation, documentation and report preparation.
This includes the original vote/linear/fusion methods, disagreement filtering,
centered features, listwise ranking, answer pooling, pairwise ranking, the small
MLP, reliability analysis and resource/reproduction scripts. The student ran
the recorded experiments locally and is responsible for understanding and
validating the code, evidence and report.

Supplied course code, numeric parser, scoring/checking utilities, data,
embeddings and references are reused. Keep all original licenses and provenance
including GSM8K and the Qwen3-0.6B attribution. No external training datasets or
additional pretrained weights were introduced. Course Lecture 6 motivated the
pairwise preference-scoring experiment. No external teacher model was used.

The initial fusion public test score (360/467) motivated further development.
The relative model was selected on validation and then tested (377/467).
Listwise, pooling, pairwise and MLP additions were not adopted and were not
publicly tested. Fresh-environment evaluation reproduced the existing final
checkpoint. The report does not claim an untouched public test bank for the
whole process. It retains both earlier and later test records and acknowledges
reused validation/training development data.
