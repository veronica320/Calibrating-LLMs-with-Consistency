'''Get the answers from multiple generated reasoning chains and save them in an additional file.'''
import math
import os
cwd = os.getcwd()
if cwd.endswith("source/calibrate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, extract_pred_answer
from itertools import product
from configuration.configuration import Config
from keys import API_KEYS, ORGANIZATION_IDS, APIKEY_ID_DICT
from model.generator import Generator
import jsonlines
from tqdm import tqdm
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test")
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
		print(f"dataset_name: {dataset_name}, LM: {LM}, output_format: {output_format}...", flush=True)
		dataset_frn = f"data/{dataset_name}/{split}.jsonl"
		dataset = load_data(dataset_frn)

		if n_vote > 1:
			n_str = f"_n:{n_vote}"
		else:
			n_str = ""
		config_name = f"{LM}_{output_format}{n_str}"

		api_key_ids = APIKEY_ID_DICT[LM]
		api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]
		org_ids = [ORGANIZATION_IDS[api_key_id] for api_key_id in api_key_ids]

		config_frn = f"source/configuration/config_files/{config_name}.json"
		config = Config.from_json_file(config_frn)
		config.split = split
		config.dataset_name = dataset_name
		config.api_keys = api_keys
		config.org_ids = org_ids
		model = Generator(config)

		preds_frn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions.jsonl"
		if not os.path.exists(preds_frn):
			print(f"predictions file {preds_frn} does not exist, skipping...")
			continue
		preds_fwn = f"output_dir/{dataset_name}/{split}/{config_name}/predictions_answers.jsonl"

		if os.path.exists(preds_fwn):
			pred_answers = load_data(preds_fwn)
			if len(pred_answers) == len(dataset):
				print(f"predictions file {preds_fwn} already exists, skipping...")
				continue

		predictions = load_data(preds_frn)

		try:
			assert len(dataset) == len(predictions)
		except:
			print(f"dataset {dataset_name} {split} has {len(dataset)} examples, but predictions file {preds_frn} has {len(predictions)} predictions, skipping...", flush=True)
			continue

		with open(preds_fwn, "w") as fw:
			writer = jsonlines.Writer(fw, flush=True)
			for i, (example, prediction) in tqdm(enumerate(zip(dataset, predictions))):
				assert int(example["id"]) == int(prediction["id"])
				example_id = int(example["id"])
				if debug and i >= 10:
					break
				if "answers" in prediction and prediction["answers"] != []:
					answers = [extract_pred_answer(dataset_name, {"answer": answer}) for answer in prediction["answers"]]
				else:
					answers = []
					if "completions" in prediction:
						completions = prediction["completions"]
					else:
						completions = prediction["reasoning_chains"]

					for completion in completions:
						answer, final_completion, _ = model.derive_answer_from_completions(example, [completion])
						processed_answer = extract_pred_answer(dataset_name, {"answer": answer})
						answers.append(processed_answer)

				# get most frequent answer except for "[invalid]"
				frequency = {}
				answer_str_to_id = {}
				for answer_id, answer in enumerate(answers):
					answer_str = str(answer)
					answer_str_to_id[answer_str] = answer_id
					if answer_str in frequency.keys():
						frequency[answer_str] += 1
					else:
						frequency[answer_str] = 1
				if "[invalid]" in frequency and len(frequency) > 1:
					del frequency["[invalid]"]

				# get the most frequent answer
				if len(frequency) == 0:
					most_frequent_answer = "[invalid]"
				else:
					most_frequent_answer_str = max(frequency, key=frequency.get)

					most_frequent_answer_id = answer_str_to_id[most_frequent_answer_str]
					most_frequent_answer = answers[most_frequent_answer_id]

				new_row = {"id": example["id"],
				           "answer": most_frequent_answer,
				           "answers": answers,
				           }

				writer.write(new_row)




