'''Generate a table of performance summary for all model configurations on all datasets in output_dir/.'''
import os
if os.getcwd().endswith("evaluation"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
import pandas as pd
from itertools import product
import csv

if __name__ == "__main__":
	LMs = ["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B",  "olmo-7B-it", "olmo-7B-it-rl"]
	prompt_names = ["standard","COT","LtM","noNL","NL+SL"]
	dataset_names = [
		"ASDIV",
		"CLUTRR",
		"DATE",
		"GSM8K",
		"MULTIARITH",
		"SAYCAN",
		"SPORT",
		"STRATEGYQA",
		"SVAMP"
	]
	ns = [40]

	performance_summary = {dataset_name: {LM: {} for LM in LMs} for dataset_name in dataset_names}

	output_dir = "output_dir"
	split = "test"

	for dataset_name, LM, n in product(dataset_names, LMs, ns):
		performance_summary[dataset_name][LM][n] = {}
		for prompt_name in prompt_names:
			n_str = f"_n:{n}" if n > 1 else ""
			score_frn = f"score_dir/answer_accuracy/{dataset_name}/{split}/{LM}_{prompt_name}{n_str}/generation_scores_predictions.jsonl.csv"
			if not os.path.exists(score_frn):
				print(f"Score file {score_frn} does not exist, skipping...")
				continue
			with open(score_frn) as f:
				score_data = pd.read_csv(score_frn).to_dict(orient="index")
				acc = float(score_data[0]['acc'])

			performance_summary[dataset_name][LM][n][prompt_name] = acc

	# save the performance summary as csv
	with open("score_dir/performance_summary.csv", "w") as fw:
		writer = csv.writer(fw)
		for LM, n in product(LMs, ns):
			writer.writerow([f"{LM}_n:{n}"])
			writer.writerow([""] + prompt_names)
			for dataset_name in dataset_names:
				accs = []
				for prompt_name in prompt_names:
					if prompt_name in performance_summary[dataset_name][LM][n]:
						acc = performance_summary[dataset_name][LM][n][prompt_name]
					else:
						acc = ""
					accs.append(acc)
				writer.writerow([dataset_name] + accs)
			writer.writerow([])