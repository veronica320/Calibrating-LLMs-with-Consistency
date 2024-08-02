'''Make predictions on the dataset using the model.'''
import os
cwd = os.getcwd()
if cwd.endswith("source/predict"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS, ORGANIZATION_IDS, APIKEY_ID_DICT
from model.generator import Generator
from model.verbalized_calibrator import VerbalizedCalibrator
from dataset.utils import load_data
import jsonlines
import time
from tqdm import tqdm
import argparse

if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", required=True)
	Parser.add_argument("--split", help="The split of the dataset.", required=True)
	Parser.add_argument("--LM", help="The name of the LM.", required=True, choices=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	Parser.add_argument("--output_format", help="The format of the output.", choices=["standard", "COT", "noNL", "NL+SL", "LtM"], required=True)
	Parser.add_argument("--n", help="The number of votes.", type=int, default=1)
	Parser.add_argument("--task_name", help="The name of the task.", choices=["generation", "calibration"], default="generation")
	Parser.add_argument("--calib_method", help="The calibration method.", choices=["ptrue", "verb_ling", "verb_percent", None])
	Parser.add_argument("--calib_shots", help="The number of exemplars to use in the calibration few-shot prompt.", type=int, default=0)
	Parser.add_argument("--completion_only", help="Only query the LM to generate the completion (reasoning chain), but not execute the solver to derive the answer.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")
	Parser.add_argument("--overwrite", help="If true, overwrite the existing predictions.", action="store_true")

	args = Parser.parse_args()
	dataset_name = args.dataset_name
	split = args.split
	LM = args.LM
	output_format = args.output_format
	n = args.n
	debug = args.debug
	completion_only = args.completion_only
	task_name = args.task_name
	calib_method = args.calib_method
	calib_shots = args.calib_shots
	overwrite = args.overwrite

	api_key_ids = APIKEY_ID_DICT[LM]
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]
	org_ids = [ORGANIZATION_IDS[api_key_id] for api_key_id in api_key_ids]

	if n > 1:
		n_str = f"_n:{n}"
	else:
		n_str = ""

	calib_str = f"_{calib_method}_{calib_shots}shot" if task_name in ["calibration"] else ""

	config_frn = f"source/configuration/config_files/{LM}_{output_format}{n_str}.json"
	config = Config.from_json_file(config_frn)
	config.dataset_name = dataset_name
	config.split = split
	config.api_keys = api_keys
	config.org_ids = org_ids
	config.task_name = task_name
	config.LM = LM
	config.calib_method = calib_method
	config.calib_shots = calib_shots

	# load the dataset, model, and specify output files
	output_dir = f"output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}"
	dataset_frn = f"data/{dataset_name}/{split}.jsonl"
	common_output_fn = None

	if task_name == "generation": # Generation stage
		answer_key = "answer"
		output_fwn = f"{output_dir}/predictions{'_completion_only' if completion_only else ''}{'_debug' if debug else ''}.jsonl"
		model = Generator(config)
		additional_data_frns = []
	else: # Calibration stage
		if task_name == "calibration":
			calib_str = f"_{calib_method}_{calib_shots}shot"
		else:
			calib_str = ""
		output_fwn = f"{output_dir}/{task_name}_predictions{calib_str}{'_debug' if debug else ''}.jsonl"
		# load additional data
		answer_key = "confidence"
		model = VerbalizedCalibrator(config)
		# load Generation predictions
		additional_data_frns = [f"{output_dir}/predictions.jsonl"]

	dataset = load_data(dataset_frn)
	if additional_data_frns is not None:
		additional_datasets = []
		for additional_data_frn in additional_data_frns:
			additional_dataset = load_data(additional_data_frn, key="id")
			additional_datasets.append(additional_dataset)
	else:
		additional_datasets = None
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# load existing predictions if any
	existing_preds = {}
	if not overwrite:
		if os.path.isfile(output_fwn):
			with open(output_fwn, "r") as fr:
				reader = jsonlines.Reader(fr)
				for line in reader:
					if "answer" in line and line["answer"] in ["[error]", "[invalid]"] or \
						answer_key in line and line[answer_key] in ["[error]", "[invalid]"]:
						continue
					example_id = line["id"]
					existing_preds[example_id] = line
			output_fwn = output_fwn.replace(".jsonl", "_contd.jsonl")
	else: # overwrite
		# save old predictions
		if os.path.isfile(output_fwn):
			os.rename(output_fwn, output_fwn.replace(".jsonl", "_old.jsonl"))

	# make predictions
	print(f"Making predictions on dataset {dataset_name} with {len(dataset)} examples, with LM {LM}, output format {output_format}, and {len(existing_preds)} existing predictions...")
	print(f"Writing predictions to {output_fwn}...")

	with open(output_fwn, 'w') as fw:
		writer = jsonlines.Writer(fw, flush=True)

		t0 = time.time()
		for i, example in tqdm(enumerate(dataset), file=sys.stdout):
			if debug and i >= 10:
				break

			question_id = int(example["id"])

			if question_id in existing_preds:
				row = existing_preds[question_id]
				writer.write(row)
				continue

			# delete the gold generation answer
			if "answer" in example:
				del example["answer"]

			# get all the information from additional data
			if additional_datasets is not None:
				additional_example = {}
				for additional_dataset in additional_datasets:
					if question_id in additional_dataset:
						additional_example.update(additional_dataset[question_id])
				example.update(additional_example)
			try:
				if task_name == "generation": # Generation stage
					example["edited_chain"] = ""
					output = model.predict(example, completion_only=completion_only)
					row = {"id": question_id,
					       "answer": output["answer"],
					       "reasoning_chain": output["completion"],
					       "answers": output["answers"],
					       "reasoning_chains": output["completions"],
					       "ppl_reciprocals": output["ppl_reciprocals"]
					       }
				else: # Calibration stage
					output = model.predict(example)
					row = {"id": question_id,
					       "confidence": output["answer"],
					       "calibration_chain": output["completion"],
					       "calibration_chains": output["completions"]}

			except Exception as e:
				row = {"id": question_id,
				       "answer": "[error]",
				       "completion": str(e),
				       "completions": ""
				       }
				print(f"Error at example {i}: {str(e)}", file=sys.stderr)

			writer.write(row)

		if i % 50 == 0:
			print(f"Finished {i} examples in {time.time() - t0} seconds.", flush=True)