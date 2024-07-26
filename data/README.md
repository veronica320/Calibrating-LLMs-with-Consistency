# Data

Each dataset has a directory under `data/`, containing the test set `test.jsonl` used for evaluation (and optionally `train.jsonl` and `dev.jsonl` if available).

In a `.jsonl` file, each line is a JSON object with the following fields:

- `id`: the unique ID of the example.
- `question`: the question.
- `answer`: the gold answer.

There are also some dataset-specific optional fields from the original dataset, e.g. `explanation` in Strategy QA. These fields are not used by our model during inference, but only kept for reference.

Please see Appendix D in our paper for details like task, statistics, source, and license.