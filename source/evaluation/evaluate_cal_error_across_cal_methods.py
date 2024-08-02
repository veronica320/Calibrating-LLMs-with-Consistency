'''
Compare calibration performance across different calibration methods:
baselines (verb_ling_0shot, verb_percent_0shot, ppl, ptrue_0shot, ptrue_8shot) and consistency metrics (entropy, agree_percent, fsd).
'''
import os
cwd = os.getcwd()
if cwd.endswith("source/evaluation"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data
from calibrate.utils import compute_correctness, compute_correlation_and_calibration_error
from itertools import product
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test", choices=["test", "dev_100"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--baselines", nargs="+", default=["verb_ling_0shot", "verb_percent_0shot", "ppl", "ptrue_0shot", "ptrue_8shot"])
	parser.add_argument("--consistency_metrics", nargs="+", default=["entropy", "agree_percent", "fsd"])
	parser.add_argument("--n_bins", type=int, default=10, help="The number of bins for computing ECE.")
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--vals", nargs="+", default=["ece", "brier"], help="The calibration error metrics to compute. Default is ece and brier. All options: correlation, p_value, ece, rms, brier, auroc, ce_correct, ce_incorrect, auprc_correct, auprc_incorrect.")
	args = parser.parse_args()

	dataset_names = args.dataset_names
	LMs = args.LMs
	output_formats = args.output_formats
	split = args.split
	n_vote = args.n_vote
	debug = args.debug
	baselines = args.baselines
	n_bins = args.n_bins
	vals = args.vals
	consistency_metrics = args.consistency_metrics
	confidence_metrics = consistency_metrics + baselines
	correctness_metric = "majority"

	if n_vote > 1:
		n_str = f"_n:{n_vote}"
	else:
		n_str = ""

	for LM, output_format in product(LMs, output_formats):
		config_name = f"{LM}_{output_format}{n_str}"
		results = {val:{
			dataset_name: {
				metric: None for metric in consistency_metrics + baselines
			} for dataset_name in dataset_names + ["average"]}
		for val in vals}

		for dataset_name in dataset_names:
			# load dataset
			dataset_frn = f"data/{dataset_name}/{split}.jsonl"
			dataset = load_data(dataset_frn)

			# load predictions
			preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions_answers.jsonl"
			if not os.path.exists(preds_frn):
				print(f"predictions file {preds_frn} does not exist, skipping...")
				continue
			predictions = load_data(preds_frn)

			# compute correctness
			correctness_list = []
			for i, (example, prediction) in enumerate(zip(dataset, predictions)):
				assert int(example["id"]) == int(prediction["id"])
				example_id = int(example["id"])
				if debug and i >= 10:
					break
				correct_binary = compute_correctness(example, prediction, dataset_name, method=correctness_metric)
				correctness_list.append(correct_binary)

			# compute calibration performance for each confidence metric
			for confidence_metric in confidence_metrics:
				baseline_frn = f"output_dir/{dataset_name}/{split}/{config_name}/calibration_predictions_{confidence_metric}.jsonl"
				if not os.path.exists(baseline_frn):
					print(f"Baseline file {baseline_frn} does not exist, skipping...")
					continue
				baseline_preds = load_data(baseline_frn)
				baseline_confidence_list = []
				for i, (example, prediction) in enumerate(zip(dataset, baseline_preds)):
					assert int(example["id"]) == int(prediction["id"])
					example_id = int(example["id"])
					if debug and i >= 10:
						break
					baseline_confidence = prediction["confidence"]
					try:
						baseline_confidence = float(baseline_confidence)
						if baseline_confidence < 0.0:
							baseline_confidence = 0.0
					except ValueError:
						baseline_confidence = 0.0
					baseline_confidence_list.append(baseline_confidence)
				try:
					assert len(baseline_confidence_list) == len(correctness_list)
				except AssertionError:
					print(f"Inconsistent number of examples for {baseline_frn}. Skipping...")
					continue
				scores = compute_correlation_and_calibration_error(baseline_confidence_list, correctness_list, n_bins=n_bins)
				for val in vals:
					if scores[val] is None:
						results[val][dataset_name][confidence_metric] = "None"
						continue
					if val == "p_value":
						results[val][dataset_name][confidence_metric] = f"{scores[val]:.3f}"
					else:
						results[val][dataset_name][confidence_metric] = round(scores[val], 3)

		# compute average metric across datasets
		for confidence_metric in confidence_metrics:
			for val in vals:
				if val == "p_value":
					results[val]["average"][confidence_metric] = ""
					continue
				vals_across_datasets = []
				for dataset_name in dataset_names:
					vals_across_datasets.append(results[val][dataset_name][confidence_metric])
				# remove None values
				vals_across_datasets = [val for val in vals_across_datasets if val and val != "None"]
				results[val]["average"][confidence_metric] = round(np.mean(vals_across_datasets), 3)

		# save results
		output_dir = f"score_dir/calibration_error/{LM}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/calerror_across_methods_{output_format}_{correctness_metric}.csv"
		df = pd.DataFrame.from_dict({(i, j): results[i][j]
									 for i in results.keys()
									 for j in results[i].keys()},
									orient='index')
		df.to_csv(fwn)