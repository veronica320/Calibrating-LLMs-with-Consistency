'''
Derive the answer from the completions generated by model.
This is useful when you need to run the prediction script on a server, where it may be non-trivial to install certain solvers (e.g. Soufflé).
In this case, you can simply run `predict.py` with the `--completion_only` flag, which will generate the completions only but not derive the answer.
Then, on your local machine with the necessary solvers installed, you can run this script to derive the answer from the completions.
'''
import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS, ORGANIZATION_IDS, APIKEY_ID_DICT
import argparse
from model.generator import Generator
from dataset.utils import load_data
import jsonlines
from tqdm import tqdm

if __name__ == "__main__":

	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.")
	Parser.add_argument("--split", help="The split of the dataset.")
	Parser.add_argument("--LM", help="The name of the LM.", required=True, choices=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	Parser.add_argument("--output_format", help="The format of the output")
	Parser.add_argument("--n", help="The number of votes", type=int, default=1)
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")

	args = Parser.parse_args()
	LM = args.LM
	output_format = args.output_format
	n = args.n
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug

	api_key_ids = APIKEY_ID_DICT[LM]
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]
	org_ids = [ORGANIZATION_IDS[api_key_id] for api_key_id in api_key_ids]

	if n > 1:
		n_str = f"_n:{n}"
	else:
		n_str = ""

	config_frn = f"source/configuration/config_files/{LM}_{output_format}{n_str}.json"
	config = Config.from_json_file(config_frn)
	config.split = split
	config.dataset_name = dataset_name
	config.api_keys = api_keys
	config.org_ids = org_ids
	model = Generator(config)

	# load the dataset, model, and specify output files
	output_dir = f"output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}"
	dataset_frn = f"data/{dataset_name}/{split}.jsonl"
	completion_frn = f"{output_dir}/predictions_completion_only{'_debug' if debug else ''}.jsonl"
	if not os.path.exists(completion_frn):
		completion_frn = f"{output_dir}/predictions{'_debug' if debug else ''}.jsonl"
	output_fwn = f"{output_dir}/predictions{'_debug' if debug else ''}.jsonl"

	# predict
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	dataset = load_data(dataset_frn)
	all_completions = load_data(completion_frn)

	existing_preds = {}
	if os.path.isfile(output_fwn):
		with open(output_fwn, "r") as fr:
			reader = jsonlines.Reader(fr)
			for line in reader:
				if "answer" in line and line["answer"] not in ["[error]", "[invalid]"]:
					example_id = int(line["id"])
					existing_preds[example_id] = line
		output_fwn = output_fwn.replace(".jsonl", "_contd.jsonl")

	print(f"Getting answers from completions {completion_frn} and writing predictions to {output_fwn}, with {len(existing_preds)} existing predictions...")
	with open(output_fwn, 'w') as fw:
		writer = jsonlines.Writer(fw, flush=True)
		for i, (example, prediction) in tqdm(enumerate(zip(dataset, all_completions))):
			assert int(example["id"]) == int(prediction["id"]), f"Example {i} has id {example['id']} but prediction {i} has id {prediction['id']}"

			if debug and i >= 10:
				break

			question_id = int(example["id"])

			if question_id in existing_preds:
				row = existing_preds[question_id]
				writer.write(row)
				continue

			# for backward compatibility
			if "completions" in prediction:
				completions = prediction["completions"]
			elif "completion" in prediction:
				completions = [prediction["completion"]]
			elif "reasoning_chains" in prediction:
				completions = prediction["reasoning_chains"]
			elif "reasoning_chain" in prediction:
				completions = [prediction["reasoning_chain"]]
			else:
				raise ValueError(f"No completion found in example {i}:\n{prediction}")
			completion_w_line_ids = completions[0] if completions else ""

			answer, final_completion, answers = model.derive_answer_from_completions(example, completions)

			prediction = {
				"id": example["id"],
				"answer": answer,
				"answers": answers,
				"reasoning_chain": completion_w_line_ids,
				"reasoning_chains": completions,
				"ppl_reciprocals": prediction["ppl_reciprocals"] if "ppl_reciprocals" in prediction else []
			}
			writer.write(prediction)

	print(f"Finished predicting on {len(dataset)+1} examples. Output written to {output_fwn}.")
