'''Reformat answer to be in the format according to "extract_pred_answer" in source/dataset/utils.py.'''
import os
if os.getcwd().endswith("predict"):
	os.chdir("../..")
import sys
sys.path.append("source")
import argparse
from dataset.utils import load_data, extract_gold_answer, extract_pred_answer
from evaluation.evaluate_answer_acc import is_correct
import jsonlines
from itertools import product

def reformat_chain(chain, config_name, dataset_name):
	'''Reformat a chain into a numbered list of steps.

	:param chain (str): The chain to be reformatted.
	:param model_name (str): The name of the model.
	:param dataset_name (str): The name of the dataset.

	:return (str): The reformatted chain.
	'''
	chain = chain.strip()
	lines = chain.split("\n")

	new_chain_with_lineid = []
	new_chain_wo_lineid = []
	line_id = 0
	for line in lines:
		line = line.strip()
		if not line:
			continue
		if any(prompt in config_name for prompt in ["NL+SL", "LtM"]): # prompts with step ids already
			new_line = line
		else:
			if dataset_name in ["SAYCAN"]:
				new_line = line
			else:
				new_line = f"{line_id + 1}. {line}"
		new_chain_with_lineid.append(new_line)
		new_chain_wo_lineid.append(line)
		line_id += 1

	new_chain_with_lineid = "\n".join(new_chain_with_lineid)
	new_chain_wo_lineid = "\n".join(new_chain_wo_lineid)
	return new_chain_with_lineid, new_chain_wo_lineid

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--split", help="The split of the dataset.")
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--debug", action="store_true", help="whether to run in debug mode")

	args = parser.parse_args()
	dataset_names = args.dataset_names
	split = args.split
	LMs = args.LMs
	output_formats = args.output_formats
	n_vote = args.n_vote
	debug = args.debug

	debug_str = "_debug" if debug else ""

	for dataset_name in dataset_names:
		# load the dataset
		dataset_frn = f"data/{dataset_name}/{split}.jsonl"
		dataset = load_data(dataset_frn)

		for LM, output_format in product(LMs, output_formats):
			if n_vote > 1:
				n_str = f"_n:{n_vote}"
			else:
				n_str = ""
			config_name = f"{LM}_{output_format}{n_str}"
			print(config_name)
			predictions_reformatted = []
			preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions{debug_str}.jsonl"
			preds_raw_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions{debug_str}_raw.jsonl"
			if not os.path.exists(preds_frn):
				print(f"File {preds_frn} doesn't exist. Skipping ...")
				continue

			if os.path.exists(preds_raw_frn):
				predictions = load_data(preds_raw_frn)
			else:
				os.rename(preds_frn, preds_raw_frn)
				predictions = load_data(preds_raw_frn)

			for example, prediction in zip(dataset, predictions):
				gold_id = int(example["id"])
				pred_id = int(prediction["id"])

				try:
					assert gold_id == pred_id
				except:
					raise AssertionError(f"Gold id {gold_id} doesn't match pred id {pred_id}.")

				gold_answer = extract_gold_answer(dataset_name, example)
				pred_answer = extract_pred_answer(dataset_name, prediction)

				if "reasoning_chain" in prediction:
					reasoning_chain = prediction["reasoning_chain"]
				else:
					reasoning_chain = prediction["completion"]

				if "reasoning_chains" in prediction:
					reasoning_chains = prediction["reasoning_chains"]
				elif "completions" in prediction:
					reasoning_chains = prediction["completions"]
				else:
					reasoning_chains = [reasoning_chain]

				reformatted_pred_chain_w_lineid, reformatted_pred_chain_wo_lineid = reformat_chain(reasoning_chain, config_name, dataset_name)
				correct = is_correct(dataset_name, gold_answer, pred_answer)

				new_prediction = {
					"id": gold_id,
					"answer": pred_answer,
					"is_correct": "Yes" if correct else "No",
					"answers": prediction["answers"] if "answers" in prediction else [],
					"reasoning_chain": reformatted_pred_chain_w_lineid,
					"reasoning_chains": reasoning_chains,
					"ppl_reciprocals": prediction["ppl_reciprocals"] if "ppl_reciprocals" in prediction else [],
				}
				predictions_reformatted.append(new_prediction)

			# save the reformatted predictions
			preds_reformatted_fwn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions{debug_str}.jsonl"
			with open(preds_reformatted_fwn, "w") as fw:
				writer = jsonlines.Writer(fw)
				writer.write_all(predictions_reformatted)








