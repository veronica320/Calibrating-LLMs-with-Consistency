'''Get confidence predictions from raw logits (reciprocal of average perplexity).'''
import os
cwd = os.getcwd()
if cwd.endswith("source/calibrate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data
from itertools import product
import jsonlines
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test", choices=["test", "dev_100"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()

	dataset_names = args.dataset_names
	LMs = args.LMs
	output_formats = args.output_formats
	split = args.split
	n_vote = args.n_vote
	debug = args.debug

	for dataset_name, LM, output_format in product(dataset_names, LMs, output_formats):
		config_name = f"{LM}_{output_format}_n:{n_vote}"
		preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions_raw.jsonl"
		if not os.path.exists(preds_frn):
			preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions.jsonl"
			if not os.path.exists(preds_frn):
				print(f"predictions file {preds_frn} does not exist, skipping...")
				continue
		predictions = load_data(preds_frn)

		fwn = f"output_dir/{dataset_name}/{split}/{config_name}/calibration_predictions_ppl.jsonl"

		print(f"Processing {preds_frn} ...")

		rows = []
		ppls_complete = True
		for pred in predictions:
			reasoning_chain = pred["reasoning_chain"] if "reasoning_chain" in pred else pred["completion"]
			reasoning_chains = pred["reasoning_chains"] if "reasoning_chains" in pred else pred["completions"]
			example_id = pred["id"]
			if "ppl_reciprocals" not in pred or pred["ppl_reciprocals"] == []:
				print(f"ppl_reciprocals not found for example {example_id} in {preds_frn}, skipping...")
				ppls_complete = False
				break
			ppl_reciprocals = pred["ppl_reciprocals"]
			# find the index of reasoning_chain in reasoning_chains
			try:
				idx = reasoning_chains.index(reasoning_chain)
				ppl = round(ppl_reciprocals[idx], 3)
			except ValueError:
				print(f"reasoning_chain not found in reasoning_chains for example {example_id} in {preds_frn}, skipping...")
				ppl = 0.0
			rows.append({"id": example_id, "confidence": ppl, "calibration_chain": "", "calibration_chains": ""})

		with open(fwn, "w") as fw:
			writer = jsonlines.Writer(fw)
			writer.write_all(rows)