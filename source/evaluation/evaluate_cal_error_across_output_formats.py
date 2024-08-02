'''
Compare calibration error across different output formats.
'''
import os
cwd = os.getcwd()
if cwd.endswith("source/evaluation"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data
from calibrate.utils import compute_correctness, compute_correlation_and_calibration_error
from calibrate.consistency import calculate_consistency
from itertools import product
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test", choices=["test", "dev_100"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--baselines", nargs="+", default=["verb_ling_0shot", "verb_percent_0shot", "ppl", "ptrue_0shot", "ptrue_8shot"])
	parser.add_argument("--consistency_metrics", nargs="+", default=["entropy", "agree_percent", "fsd"])
	parser.add_argument("--n_bins", type=int, default=10)
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--vals", nargs="+", default=["ece", "brier"], help="The calibration error metrics to compute.")
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
	correctness_metric = "majority"

	for LM, consistency_metric in product(LMs, consistency_metrics):
		print(f"LM: {LM}, consistency_metric: {consistency_metric}, correctness_metric: {correctness_metric} ...")
		results = {val:{dataset_name: {
			output_format: None for output_format in output_formats
		} for dataset_name in dataset_names + ["average"]} for val in vals}

		# create plot for
		for dataset_name, output_format in product(dataset_names, output_formats):

			dataset_frn = f"data/{dataset_name}/{split}.jsonl"
			dataset = load_data(dataset_frn)

			config_name = f"{LM}_{output_format}_n:{n_vote}"

			preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions_answers.jsonl"
			if not os.path.exists(preds_frn):
				print(f"predictions file {preds_frn} does not exist, skipping...")
				continue
			predictions = load_data(preds_frn)

			consistency_list = []
			correctness_list = []

			for i, (example, prediction) in enumerate(zip(dataset, predictions)):
				assert int(example["id"]) == int(prediction["id"])
				example_id = int(example["id"])
				if debug and i >= 10:
					break

				answers = prediction["answers"]
				pred_answer = prediction["answer"]

				consistency = calculate_consistency(answers, method=consistency_metric)

				correct_binary = compute_correctness(example, prediction, dataset_name, method=correctness_metric)

				consistency_list.append(consistency)
				correctness_list.append(correct_binary)

			# compute correlation and calibration error metrics
			scores = compute_correlation_and_calibration_error(consistency_list, correctness_list, n_bins=n_bins)
			for val in vals:
				if scores[val] is None:
					results[val][dataset_name][output_format] = None
					continue
				if val == "p_value":
					results[val][dataset_name][output_format] = f"{scores[val]:.3f}"
				else:
					results[val][dataset_name][output_format] = round(scores[val], 3)

		for output_format in output_formats:
			# compute average across datasets
			for val in vals:
				if val == "p_value":
					results[val]["average"][output_format] = ""
					continue
				vals_across_datasets = []
				for dataset_name in dataset_names:
					vals_across_datasets.append(results[val][dataset_name][output_format])
				# delete None values
				vals_across_datasets = [val for val in vals_across_datasets if val is not None]
				try:
					results[val]["average"][output_format] = round(np.mean(vals_across_datasets), 3)
				except:
					print(vals_across_datasets)

		output_dir = f"score_dir/calibration_error/{LM}"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/calerror_across_formats_{consistency_metric}_{correctness_metric}.csv"

		# save results to csv
		# use the order of vals as the order of row groups
		df = pd.DataFrame.from_dict({(i, j): results[i][j]
									 for i in vals
									 for j in results[i].keys()},
									orient='index')
		# Reset index to make the multi-level index columns
		df.reset_index(inplace=True)
		# Rename columns to match the CSV file structure
		df.columns = ['Metric', 'Dataset'] + output_formats
		# Sort the DataFrame based on the 'vals' list while keeping the category order
		df['Metric'] = pd.Categorical(df['Metric'], categories=vals, ordered=True)
		df_sorted = df.sort_values(by=['Metric'])

		df_sorted['Dataset'] = pd.Categorical(df_sorted['Dataset'], categories=dataset_names+["average"], ordered=True)
		df_final_sorted = df_sorted.sort_values(by=['Metric', 'Dataset'])

		# Display the first few rows of the sorted DataFrame
		df_final_sorted.head()
		df_final_sorted.to_csv(fwn, index=False)
