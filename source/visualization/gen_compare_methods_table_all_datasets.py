'''
Generate the table to compare calibration performance of all calibration methods,
where scores are averaged across output formats for each LM.
'''
import os
cwd = os.getcwd()
if cwd.endswith("source/visualization"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-instruct"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test", choices=["test", "dev_100"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--baselines", nargs="+", default=["verb_ling_0shot", "verb_percent_0shot", "ppl", "ptrue_0shot", "ptrue_8shot"])
	parser.add_argument("--consistency_metrics", nargs="+", default=["entropy", "agree_percent", "fsd"])
	parser.add_argument("--n_bins", type=int, default=10)
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--vals", nargs="+", default=["ece", "brier"], help="The calibration error metrics to compute. Options: correlation, p_value, ece, rms, brier, auroc, ce_correct, ce_incorrect, auprc_correct, auprc_incorrect")
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

	for val in vals:
		all_results = {
			LM: {
				dataset:{
					confidence_metric: "n/a" for confidence_metric in confidence_metrics
				} for dataset in dataset_names + ["average"]
			} for LM in LMs
		}
		for LM in LMs:
			LM_results = {
				dataset:{
					confidence_metric: {
						output_format: None for output_format in output_formats
					} for confidence_metric in confidence_metrics
				} for dataset in dataset_names + ["average"]
            }
			for output_format in output_formats:
				config_name = f"{LM}_{output_format}{n_str}"
				output_dir = f"score_dir/calibration_error/{LM}"
				frn = f"{output_dir}/calerror_across_methods_{output_format}_{correctness_metric}.csv"
				results = pd.read_csv(frn, index_col=[0, 1]).to_dict()

				for confidence_metric in confidence_metrics:
					for dataset in dataset_names:
						LM_results[dataset][confidence_metric][output_format] = "{:.3f}".format(results[confidence_metric][(val, dataset)])
					# average across datasets
					LM_results["average"][confidence_metric][output_format] = "{:.3f}".format(results[confidence_metric][(val, "average")])
			# compute average across output formats
			for dataset in dataset_names + ["average"]:
				for confidence_metric in confidence_metrics:
					all_results[LM][dataset][confidence_metric] = "{:.3f}".format(np.mean([float(v) for v in LM_results[dataset][confidence_metric].values()]))
		# save results
		output_dir = f"score_dir/calibration_error/all_LMs"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/calerror_across_methods_{val}_all_datasets.csv"
		df = pd.DataFrame.from_dict({(i, j): all_results[i][j]
		                             for i in all_results.keys()
		                             for j in all_results[i].keys()},
		                            orient="index")
		df.to_csv(fwn)