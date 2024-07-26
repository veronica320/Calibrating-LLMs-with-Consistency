'''
Generate the table to compare calibration performance of all calibration methods,
where scores are averaged across all datasets and output formats for each LM.
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

	all_results = {
		val: {
			LM:{
				confidence_metric: None for confidence_metric in confidence_metrics
			} for LM in LMs
		} for val in vals
	}
	for val in vals:
		for LM in LMs:
			LM_results = {
				confidence_metric: {
					output_format: None for output_format in output_formats
				} for confidence_metric in confidence_metrics
			}
			for output_format in output_formats:
				config_name = f"{LM}_{output_format}{n_str}"
				output_dir = f"score_dir/calibration_error/{LM}"
				frn = f"{output_dir}/calerror_across_methods_{output_format}_{correctness_metric}.csv"
				results = pd.read_csv(frn, index_col=[0, 1]).to_dict()
				# convert to dict
				for confidence_metric in confidence_metrics:
					LM_results[confidence_metric][output_format] = results[confidence_metric][(val, "average")]
			# compute average across output formats
			for confidence_metric in confidence_metrics:
				all_results[val][LM][confidence_metric] = round(np.mean(list(LM_results[confidence_metric].values())), 3)

		# save results
		output_dir = f"score_dir/calibration_error/all_LMs"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		fwn = f"{output_dir}/calerror_across_methods_{val}_avg.csv"
		df = pd.DataFrame(all_results[val])
		# switch rows and columns
		df = df.transpose()
		df.to_csv(fwn)