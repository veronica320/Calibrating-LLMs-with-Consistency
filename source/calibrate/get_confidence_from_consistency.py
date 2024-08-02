'''Get confidence predictions from three possible consistency metrics: entropy, agree_percent, fsd.'''
import os
cwd = os.getcwd()
if cwd.endswith("source/calibrate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data
from consistency import calculate_consistency
from itertools import product
import jsonlines
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test")
	parser.add_argument("--consistency_metrics", nargs="+", default=["entropy", "agree_percent", "fsd"])
	parser.add_argument("--n_vote", type=int, default=40)
	args = parser.parse_args()

	dataset_names = args.dataset_names
	LMs = args.LMs
	output_formats = args.output_formats
	split = args.split
	consistency_metrics = args.consistency_metrics
	n_vote = args.n_vote

	if n_vote > 1:
		n_str = f"_n:{n_vote}"
	else:
		n_str = ""

	for dataset_name, LM, output_format in product(dataset_names, LMs, output_formats):
		print(f"Processing {dataset_name} {LM} {output_format} n:{n_vote} ...")
		config_name = f"{LM}_{output_format}{n_str}"

		preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions_answers.jsonl"
		if not os.path.exists(preds_frn):
			print(f"predictions file {preds_frn} does not exist, skipping...")
			continue
		predictions = load_data(preds_frn)

		for consistency_metric in consistency_metrics:
			fwn = f"output_dir/{dataset_name}/{split}/{config_name}/calibration_predictions_{consistency_metric}.jsonl"
			rows = []
			for prediction in predictions:
				example_id = prediction["id"]
				answers = prediction["answers"]
				consistency = calculate_consistency(answers, method=consistency_metric)
				rows.append({"id": example_id, "confidence": consistency, "calibration_chain": "", "calibration_chains": ""})

			with open(fwn, "w") as fw:
				writer = jsonlines.Writer(fw)
				writer.write_all(rows)