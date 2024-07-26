# Calibrating-LLMs-with-Consistency
Code and data accompanying our paper ["Calibrating Large Language Models with Sample Consistency"](https://arxiv.org/abs/2402.13904) on arxiv.

## Table of Contents
- [Get started](#get-started)
- [Repo Structure](#repo-structure)
- [Usage](#usage)
  - [Make predictions](#make-predictions)
  - [Evaluate prediction accuracy](#evaluate-prediction-accuracy)
  - [Calibrate prediction confidence](#calibrate-prediction-confidence)
  - [Evaluate calibration error](#evaluate-calibration-error)
  - [Visualization](#visualization)
- [Citation](#citation)

## Get started
We suggest using miniconda/conda to set up the environment. The `environment.yml` file specifies the minimal dependencies. You can create a virtual environment by running:

```
cd /path/to/Calibrating-LLMs-with-Consistency
conda env create -p ./envs -f environment.yml python=3.9
```

Additionally, to run experiments on **StrategyQA**, you should install [Soufflé](https://souffle-lang.github.io/index.html) (a version of Datalog interpreter we use) following [these instructions](https://souffle-lang.github.io/build). It's not a Python package, so you'll need to install it separately. Note that under the "Installing Souffle" section you should use `-DCMAKE_INSTALL_PREFIX="~/.local"` for it to be installed to the right place. 

## Repo Structure
```
Calibrating-LLMs-with-Consistency/
├── data/ # Datasets for evaluation. See `data/README.md` for details.
│ 
├── source/ # Source code.
│ ├── configuration/
│ │ ├── configuration.py # Config class. See the definition of each field in the init() funciton. 
│ │ └── config_files/ # Configuration files for each model. Each file name is in the format of `{model-name}.json`, e.g., `code002_NL+SL.json`. See `configuration/README.md` for details.
│ │ 
│ ├── dataset/ # Dataset utility functions.
│ │ 
│ ├── evaluation/ # Evaluation scripts.
│ │ ├── evaluate_answer_acc.py # Script to evaluate prediction accuracy.
│ │ ├── evaluate_cal_error_across_cal_methods.py.py # Script to evaluate and compare calibration errors across different different calibration methods.
│ │ └── evaluate_cal_error_across_output_formats.py # Script to evaluate and compare calibration errors across different output formats.
│ │ 
│ ├── predict/ # Prediction scripts.
│ │ ├── predict.py # Script to make predictions using the models.
│ │ ├── get_answer_from_completion.py # Derive answers from completions.
│ │ ├── reformat_answer.py # Reformat predictions in target format.
│ │ └── logs/ # Logs from prediction process.
│ │ 
│ ├── model/ # Model classes.
│ │ ├── generator.py # Generator class for predicting answers.
│ │ ├── verbalized_calibrator.py # Verbalized calibrator class for calibrating predictions.
│ │ └── solver/ # Solver functions for deriving answers for different datasets.
│ │ 
│ ├── prompt/ # Prompts for in-context learning. See `source/prompt/README.md` for details.
│ │ 
│ └── visualization/ # Scripts to generate tables and plots in the paper.
│ 
├── output_dir/ # Output directory for model predictions.
├── environment.yml # Conda environment file.
└── README.md # This file.
```

## Usage

### Make predictions

1. Provide your OpenAI API key(s) by creating a file called `key.py` under `source/` in the following format:
```
API_KEYS = {
	"key1_nickname": "key1",
	"key2_nickname": "key2",
	...
}
```
Note that your keys should have access to the relevant LM (`gpt-3.5-turbo`, etc.) specified in the configuration you'd like to use.

2. Choose a model configuration you'd like to use. You can use an existing configuration under `configuration/config_files/{dataset_name}` or create a new one. See `configuration/README.md` for details.

3. Run `source/predict/predict.py`:
```
$ python predict.py -h
usage: predict.py [-h] --dataset_name DATASET_NAME --split SPLIT --LM LM
                  --output_format {standard,COT,noNL,NL+SL,LtM} [--n N]
                  [--task_name {generation}]
                  [--completion_only] [--debug]
                  [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset.
  --split SPLIT         The split of the dataset.
  --LM LM               The name of the LM.
  --output_format {standard,COT,noNL,NL+SL,LtM}
                        The format of the output.
  --n N                 The number of votes.
  --task_name {generation}
                        The name of the task.
  --completion_only     Only query the LM to generate the completion (reasoning
                        chain), but not execute the solver to derive the answer.
  --debug               If true, only run on the first 10 examples.
  --overwrite           If true, overwrite the existing predictions.
```


Example:
```
nohup python predict.py --dataset_name "GSM8K" --split "test" --LM "gpt4" --output_format "COT" --n 40 > logs/GSM8K/gpt4_COT_test_n:40.log 2>&1 &
```

The model predictions will be saved under `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}`, for instance, `output_dir/GSM8K/test/gpt4_COT_n:40/predictions.jsonl` in the above example. 
See `output_dir/README.md` for details on the format.

Tips: 
- It's recommended to use `nohup` since certain experiments can take hours to run. Also, you may need to create the relevant `logs/{dataset_name}` directory if it doesn't exist.
- The `--completion_only` flag is useful when you run the prediction script on a server, where it may be non-trivial to install certain solvers (e.g. Soufflé). In this case, you can simply run `predict.py` with the `--completion_only` flag on, which will generate the completions only but not derive the answer. Then, on your local machine with the necessary solvers installed, you can run `source/predict/get_answer_from_completion.py` (with the same arguments) to derive the answer from the completions.

### Evaluate prediction accuracy
Run `source/evaluation/evaluate_answer_acc.py` with the following arguments:
```
$ python evaluate_answer_acc.py -h
usage: evaluate_answer_acc.py [-h] --dataset_name DATASET_NAME --split SPLIT
                              --preds_fn PREDS_FN --LM
                              {code002,gpt-3.5-turbo,gpt4,llama-7B,llama-13B,llama-70B,mistral-7B,mistral-7B-instruct,olmo-7B,olmo-7B-instruct,olmo-7B-instruct-rlhf}
                              --output_format {standard,COT,noNL,NL+SL,LtM}
                              [--n_vote N_VOTE] [--non_empty_only] [--valid_only]
                              [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset.
  --split SPLIT         The split of the dataset.
  --preds_fn PREDS_FN   The name of the predictions file.
  --LM {code002,gpt-3.5-turbo,gpt4,llama-7B,llama-13B,llama-70B,mistral-7B,mistral-7B-instruct,olmo-7B,olmo-7B-instruct,olmo-7B-instruct-rlhf}
                        The name of the LM.
  --output_format {standard,COT,noNL,NL+SL,LtM}
                        The format of the output.
  --n_vote N_VOTE       The number of votes
  --non_empty_only      If true, only evaluate on non-empty answers.
  --valid_only          If true, only evaluate on valid answers.
  --debug               If true, only run on the first 10 examples.
```

Example:
```
python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM gpt4 --output_format COT --n_vote 40 --preds_fn predictions.jsonl
```
Output:
```
Dataset: GSM8K
Split: test
Model: gpt4_COT_n:40
Preds frn:output_dir/GSM8K/test/gpt4_COT_n:40/predictions.jsonl

          acc
overall  95.1
```
The accuracy will also be saved to a file called `score_dir/answer_accuracy/{dataset_name}/{split}/{LM}_{output_format}{n_str}/generation_scores_{preds_fn}.csv`, 
for instance, `score_dir/answer_accuracy/GSM8K/test/gpt4_COT_n:40/generation_scores_predictions.jsonl.csv` in the above example.


### Calibrate prediction confidence

We compare our method, [consistency-based calibration](#consistency-based-calibration), with several baseline calibration methods, including [logit-based calibration](#logit-based-calibration), as well as [P(True) and verbalized calibration](#ptrue-and-verbalized-calibration). 
After the [prerequisites](#prerequisites), you can jump to the corresponding section for more details on whichever method you choose.

#### Prerequisites
Before running any calibration method, you need to reformat the predictions by running `source/predict/reformat_answer.py` with the following arguments:
```
$ python reformat_answer.py -h
usage: reformat_answer.py [-h] [--dataset_names DATASET_NAMES [DATASET_NAMES ...]]
                          [--split SPLIT] [--LMs LMS [LMS ...]]
                          [--output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]]
                          [--n_vote N_VOTE] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_names DATASET_NAMES [DATASET_NAMES ...]
  --split SPLIT         The split of the dataset.
  --LMs LMS [LMS ...]
  --output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]
  --n_vote N_VOTE
  --debug               whether to run in debug mode
```

Example:
```
python reformat_answer.py --split test --n_vote 40
```

This will reformat the predictions for all LMs, datasets, and output formats, where the predictions (n=40) are available, on the test split.

The reformatted predictions will be saved to `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/predictions.jsonl` for each dataset, LM, and output format. The original predictions will be saved to `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/predictions_raw.jsonl`.


#### Consistency-based calibration
This is our proposed calibration method using three consistency metrics: agreement, entropy, and first-second distance (FSD).

To use this method, run `source/calibrate/get_confidence_from_consistency.py` with the following arguments:
```
$ python get_confidence_from_consistency.py -h
usage: get_confidence_from_consistency.py [-h]
                                          [--dataset_names DATASET_NAMES [DATASET_NAMES ...]]
                                          [--LMs LMS [LMS ...]]
                                          [--output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]]
                                          [--split {test,dev_100}]
                                          [--consistency_metrics CONSISTENCY_METRICS [CONSISTENCY_METRICS ...]]
                                          [--n_vote N_VOTE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_names DATASET_NAMES [DATASET_NAMES ...]
  --LMs LMS [LMS ...]
  --output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]
  --split {test,dev_100}
  --consistency_metrics CONSISTENCY_METRICS [CONSISTENCY_METRICS ...]
  --n_vote N_VOTE
```

Example:
```
python get_confidence_from_consistency.py --split test --n_vote 40
```

This will run consistency-based calibration on all LMs, datasets, and output formats, where the predictions (n=40) are available, on the test split, with all three consistency metrics.

The results will be saved to `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/calibration_predictions_{consistency_metric}.jsonl` for each dataset, LM, and output format.

#### Logit-based calibration
This is a baseline calibration method based on the probability logits of the generation. Specifically, confidence is computed as the exponential of the average log probability of all tokens in the output sequence, which is equivalent to the reciprocal of perplexity.

Since the perplexity reciprocals have already been saved when running `predict.py` to generate predictions, you can simply run `source/calibrate/get_confidence_from_logit.py` with the following arguments to get the calibration results:
```
$ python get_confidence_from_ppls.py -h
usage: get_confidence_from_ppls.py [-h]
                                   [--dataset_names DATASET_NAMES [DATASET_NAMES ...]]
                                   [--LMs LMS [LMS ...]]
                                   [--output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]]
                                   [--split {test,dev_100}] [--n_vote N_VOTE]
                                   [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_names DATASET_NAMES [DATASET_NAMES ...]
  --LMs LMS [LMS ...]
  --output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]
  --split {test,dev_100}
  --n_vote N_VOTE
  --debug
```

Example:
```
python get_confidence_from_ppls.py --split test --n_vote 40
```

This will run logit-based calibration on all LMs, datasets, and output formats, where the predictions (n=40) are available, on the test split.

The results will be saved to `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/calibration_predictions_ppl.jsonl`.

#### P(True) and verbalized calibration

These are all baseline calibration methods which prompt the LM to judege its own generation. 
Specifically, P(True) asks the LM if its generation is True or False and considers the normalized probability assigned to the ‘True’ token as its confidence (`P(True)`). Verbalized calibration directly asks the LM to generate the confidence score, either as a percentage (`verbalized percentage`) or linguistic expression then mapped to a percentage (`verbalized linguistic expression`).  

To use any of these methods, run `source/predict/predict.py` with the following arguments:
```
$ python predict.py -h
usage: predict.py [-h] --dataset_name DATASET_NAME --split SPLIT --LM
                  {code002,gpt-3.5-turbo,gpt4,llama-7B,llama-13B,llama-70B,mistral-7B,mistral-7B-instruct,olmo-7B,olmo-7B-instruct,olmo-7B-instruct-rlhf}
                  --output_format {standard,COT,noNL,NL+SL,LtM} [--n N]
                  [--task_name {calibration}]
                  [--calib_method {ptrue,verb_ling,verb_percent,None}]
                  [--calib_shots CALIB_SHOTS] [--debug]
                  [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset.
  --split SPLIT         The split of the dataset.
  --LM {code002,gpt-3.5-turbo,gpt4,llama-7B,llama-13B,llama-70B,mistral-7B,mistral-7B-instruct,olmo-7B,olmo-7B-instruct,olmo-7B-instruct-rlhf}
                        The name of the LM.
  --output_format {standard,COT,noNL,NL+SL,LtM}
                        The format of the output.
  --n N                 The number of votes.
  --task_name {calibration}
                        The name of the task.
  --calib_method {ptrue,verb_ling,verb_percent,None}
                        The calibration method.
  --calib_shots CALIB_SHOTS
                        The number of exemplars to use in the calibration few-shot
                        prompt.
  --debug               If true, only run on the first 10 examples.
  --overwrite           If true, overwrite the existing predictions.  

```
We only support `calib_shots=0` or `calib_shots=8` for P(True), and `calib_shots=0` for verbalized calibration.

Meanwhile, `gpt-3.5-turbo` and `gpt4` only support verbalized calibration but not P(True), since they don't provide the probability logtis for top-k tokens.

Example:
```
nohup python predict.py --dataset_name "GSM8K" --split "test" --LM "gpt4" --output_format "COT" --n 40 --task_name "calibration" --calib_method "verb_ling" --calib_shots 0 > logs/GSM8K/gpt4_COT_test_n:40_verb_ling_0shot.log 2>&1 &
```
The calibration results will be saved to `output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/calibration_predictions_{calib_method}_{calib_shots}shot.jsonl`.
For instance, `output_dir/GSM8K/test/gpt4_COT_n:40/calibration_predictions_verb_ling_0shot.jsonl` in the above example.

### Evaluate calibration error
In the previous step, we have obtained the calibration predictions using several calibration methods (`output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}/calibration_predictions_{calib_method}.jsonl`). Now, we can evaluate and compare the calibration error of each calibration method.

To do this, run `source/evaluation/evaluate_cal_error_across_cal_methods.py` with the following arguments:
```
$ python evaluate_cal_error_across_cal_methods.py  -h
usage: evaluate_cal_error_across_cal_methods.py [-h]
                                                [--dataset_names DATASET_NAMES [DATASET_NAMES ...]]
                                                [--LMs LMS [LMS ...]]
                                                [--output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]]
                                                [--split {test,dev_100}]
                                                [--n_vote N_VOTE]
                                                [--baselines BASELINES [BASELINES ...]]
                                                [--consistency_metrics CONSISTENCY_METRICS [CONSISTENCY_METRICS ...]]
                                                [--n_bins N_BINS] [--debug]
                                                [--vals VALS [VALS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_names DATASET_NAMES [DATASET_NAMES ...]
  --LMs LMS [LMS ...]
  --output_formats OUTPUT_FORMATS [OUTPUT_FORMATS ...]
  --split {test,dev_100}
  --n_vote N_VOTE
  --baselines BASELINES [BASELINES ...]
  --consistency_metrics CONSISTENCY_METRICS [CONSISTENCY_METRICS ...]
  --n_bins N_BINS       The number of bins for computing ECE.
  --debug
  --vals VALS [VALS ...]
                        The calibration error metrics to compute. Default is ece and
                        brier. All options: correlation, p_value, ece, rms, brier,
                        auroc, ce_correct, ce_incorrect, auprc_correct,
                        auprc_incorrect.
```

Example:
```
python evaluate_cal_error_across_cal_methods.py --split test --n_vote 40 
```

This will evaluate the calibration error of all calibration methods on all LMs, datasets, and output formats, where the calibration predictions (n=40) are available, on the test split. The results will be saved to `score_dir/calibration_error/{LM}/calerror_across_methods_{output_format}_majority.csv` for each LM and output format.

Instead of comparing across *calibration methods*, you can also compare the calibration error across *output formats* by running `source/evaluation/evaluate_cal_error_across_output_formats.py` with the same arguments. The results will be saved to `score_dir/calibration_error/{LM}/calerror_across_formats_{consistency_metric}_majority.csv` for each LM and consistency metric.

### Visualization

To generate the tables and plots in the paper, you can run the scripts under `source/visualization/`.
- To generate Tables 1 and 2, run `source/visualization/gen_compare_methods_table.py` with the default arguments. The tables will be saved to `score_dir/calibration_error/all_LMs/calerror_across_methods_brier_avg.csv`.
- To generate Figure 3, run `source/visualization/gen_compare_output_formats_figure.py` with the default arguments. The figure will be saved to `score_dir/calibration_error/all_LMs/calerror_across_output_formats_brier_deep.pdf`.
- The other scripts generate tables in the Appendix. 

## Citation
If you find this repository useful, please cite our paper:
```
@article{lyu2024calibratinglargelanguagemodels,
         title={Calibrating Large Language Models with Sample Consistency}, 
         author={Qing Lyu and Kumar Shridhar and Chaitanya Malaviya and Li Zhang and Yanai Elazar and Niket Tandon and Marianna Apidianaki and Mrinmaya Sachan and Chris Callison-Burch},
         journal={arXiv preprint arXiv:2402.13904},
         year={2024},
         url={https://arxiv.org/abs/2402.13904} 
}
```

## Funding Acknowledgements
This research is based upon work supported in part by the Air Force Research Laboratory (contract FA8750-23-C-0507), the DARPA KAIROS Program (contract FA8750-19-2-1004), the IARPA HIATUS Program (contract 2022-22072200005), the NSF (Award 1928631), the Swiss National Science Foundation (Project No. 197155), and a Responsible AI grant by the Haslerstiftung. Approved for Public Release, Distribution Unlimited. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of AFRL, DARPA, IARPA, NSF, SNSF or the U.S. Government. 






