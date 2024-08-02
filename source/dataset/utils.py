'''Dataset utilities.'''

import json
import re
import csv
from collections import Counter
from fractions import Fraction
import math

INVALID_ANS = "[invalid]"

LM_MAX_TOKEN_DICT= {
	"gpt-4": 8192,
	"gpt-4-0613": 8192,
	"gpt-4-0314": 8192,
	"code-davinci-002": 8001,
	"gpt-3.5-turbo-16k": 16385,
	"gpt-3.5-turbo-16k-0613": 16385,
	}

LM_NAME_TO_SHORT_NAME = {
	"gpt-4-0613": "gpt4",
	"code-davinci-002": "code002",
	"gpt-3.5-turbo-16k-0613": "gpt-3.5-turbo",
	"llama-7B": "llama-7B",
	"llama-13B": "llama-13B",
	"llama-70B": "llama-70B",
	"mistral-7B": "mistral-7B",
	"mistral-7B-it": "mistral-7B-it",
	"olmo-7B": "olmo-7B",
}

LM_SHORT_NAME_TO_NAME = {
	"gpt4": "gpt-4-0613",
	"code002": "code-davinci-002",
	"gpt-3.5-turbo": "gpt-3.5-turbo-16k-0613",
	"llama-7B": "llama-7B",
	"llama-13B": "llama-13B",
	"llama-70B": "llama-70B",
	"mistral-7B": "mistral-7B",
	"mistral-7B-it": "mistral-7B-it",
	"olmo-7B": "olmo-7B",
}



CALIB_STOP_TOKEN_DICT = {
	"GSM8K": "Q:",
	"CLUTRR": "Context:",
	"DATE": "Q:",
	"SVAMP": "Q:",
	"MULTIARITH": "Q:",
	"ASDIV": "Q:",
	"STRATEGYQA": "Q:",
	"SPORT": "Q:",
	"SAYCAN": "User query:",
}


# Stop token for each dataset (faithful COT prompt)
CODE_STOP_TOKEN = {"GSM8K": "# Q:",
                   "SVAMP": "# Q:",
                   "MULTIARITH": "# Q:",
                   "ASDIV": "# Q:",
                   "SAYCAN": "User query:",
                   "STRATEGYQA": "// Q:",
                   "SPORT": "# Q:",
                   "CLUTRR": "Context:",
                   "DATE": "# Q:"
                   }

# Max number of tokens needed for each dataset (faithful COT prompt)
CODE_MAX_TOKEN = { "GSM8K": 800,
                   "SVAMP": 800,
                   "MULTIARITH": 800,
                   "ASDIV": 800,
                   "SAYCAN": 300,
                   "STRATEGYQA": 800,
                   "DATE": 800,
                   "SPORT": 500,
				   "CLUTRR": 100 # for each step
                  }

NO_CODE_STOP_TOKEN = {"GSM8K": "Q:",
                      "SVAMP": "Q:",
                      "MULTIARITH": "Q:",
                      "ASDIV": "Q:",
                      "STRATEGYQA": "Q:",
                      "SPORT": "Q:",
                      "SAYCAN": "User query:",
                      "CLUTRR": "Context:",
                      "DATE": "Q:"
                      }

NO_CODE_MAX_TOKEN = {"GSM8K": 500,
                     "SVAMP": 500,
                     "MULTIARITH": 500,
                     "ASDIV": 500,
                     "SAYCAN": 300,
                     "SPORT": 500,
                     "STRATEGYQA": 500,
                     "CLUTRR": 100, # for each step
                     "DATE": 500
                     }


def load_data(frn, key=None):
	'''Load data from a file.
	:param frn (str): The dataset file name.

	:return: The dataset (a list of examples, each as a dictionary).
	'''
	if frn.endswith(".jsonl"):
		with open(frn, 'r') as fr:
			if key:
				lines = {}
				for i, line in enumerate(fr):
					if line.strip() == "":
						continue
					try:
						line_data = json.loads(line)
						lines[line_data[key]] = line_data
					except json.decoder.JSONDecodeError as e:
						print(f"Error in line {i}: {line}\n {e}")
						exit(-1)
			else:
				lines = []
				for i, line in enumerate(fr):
					if line.strip() == "":
						continue
					try:
						lines.append(json.loads(line))
					except json.decoder.JSONDecodeError as e:
						print(f"Error in line {i}: {line}\n {e}")
						exit(-1)
		return lines
	elif frn.endswith(".csv"):
		with open(frn) as fr:
			reader = csv.DictReader(fr)
			if key:
				lines = {}
				for line in reader:
					lines[line[key]] = line
			else:
				lines = [line for line in reader]
			return lines

def str2num(answer_str, rounding="int", abs_val=True):
	'''Convert a string to a number.
	@:param answer_str (str): The string to convert.
	@:param rounding (str): The rounding method for the answer. Can be "int", "ceil", or "floor".
	@:param abs_val (bool): Whether to take the absolute value of the answer.

	@:return The converted number.
	'''
	if "/" in answer_str:
		answer_str =  float(sum(Fraction(s) for s in answer_str.split()))
	answer_str = float(answer_str)

	if rounding == "int":
		answer_str = int(answer_str)
	elif rounding == "ceil":
		answer_str = math.ceil(answer_str)
	elif rounding == "floor":
		answer_str = math.floor(answer_str)
	if abs_val:
		answer_str = abs(answer_str)

	return answer_str

def extract_gold_answer(dataset_name, example):
	'''Extract the gold answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param example (dict): The gold example.

	:return: The gold answer.
	'''
	if dataset_name in ["GSM8K", "SVAMP", "MULTIARITH"]:
		gold_completion = example["answer"]
		ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
		match = ANS_RE.search(gold_completion)
		if match:
			match_str = match.group(1).strip()
			match_str = match_str.replace(",", "")
			return int(match_str)
		else:
			return INVALID_ANS
	elif dataset_name == "ASDIV":
		gold_completion = example["answer"]
		# ASDiv has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(gold_completion) in [tuple, list]: # the gold answer has multiple values
			answer = dict(Counter([int(ans) for ans in gold_completion]))
		else: # the gold answer has a single value
			answer = int(gold_completion)
		return answer
	elif dataset_name in ["CLUTRR"]:
		gold_completion = example["answer"]
		answer = gold_completion.split("#### ")[-1]
		return answer
	elif dataset_name == "SAYCAN":
		gold_completion = example["answer"]
		answer = eval(gold_completion)
		return answer
	elif dataset_name in ["STRATEGYQA"]:
		gold_completion = example["answer"]
		answer = bool(gold_completion)
		return answer
	elif dataset_name in ["SPORT"]:
		gold_completion = example["answer"]
		try:
			answer = bool(int(gold_completion))
		except:
			answer = "[error]"
		return answer
	else:
		if "answer" not in example:
			# raise KeyError(f"Dataset {dataset_name} does not have a gold answer in the example: {example}")
			return None
		else:
			return example["answer"]


def extract_pred_answer(dataset_name, prediction, rounding="int", abs_val=True):
	'''Extract the predicted answer from a completion.
	:param dataset_name (str): The name of the dataset.
	:param pred_completion (str): The predicted completion.
	:param rounding (str): The rounding method for the predicted answer. Can be "int", "ceil", or "floor".
	:param abs_val (bool): Whether to take the absolute value of the predicted answer.

	:return: The predicted answer.
	'''
	pred_completion = prediction["answer"]
	if INVALID_ANS in str(pred_completion):
		return INVALID_ANS

	if dataset_name in ["GSM8K", "SVAMP", "MULTIARITH"]:
		# GSM8K, SVAMP, and MULTIARITH all have a single-value integer answer
		if type(pred_completion) == int:
			pred_answer = pred_completion
		elif type(pred_completion) == str:
			ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
			match = ANS_RE.findall(pred_completion)
			if match:
				# get last group
				match_str = match[-1].strip()
				match_str = match_str.replace(",", "")
				try:
					pred_answer = str2num(match_str, rounding, abs_val)
				except:
					pred_answer = INVALID_ANS
			else:
				pred_answer = INVALID_ANS
		return pred_answer

	elif dataset_name in ["ASDIV"]:
		# ASDIV has questions with multi-value answers, e.g., Q: "What the width and the height of xxx?", A: (5, 10)
		if type(pred_completion) == int:
			return pred_completion
		elif type(pred_completion) == str:
			pred_completion = pred_completion.lstrip("{([").rstrip("]})")
			pred_answers = pred_completion.split(",")
			final_pred_answers = []
			for pred_answer in pred_answers:
				pred_answer = pred_answer.strip().split(":")[-1].strip("'\"")
				ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
				match = ANS_RE.findall(pred_answer)
				if match:
					# get last group
					match_str = match[-1].strip()
					match_str = match_str.replace(",", "")
					try:
						pred_answer = str2num(match_str, rounding, abs_val)
						final_pred_answers.append(pred_answer)
					except:
						continue
			if len(final_pred_answers) > 1:
				return dict(Counter(final_pred_answers))
			elif len(final_pred_answers) == 1:
				return final_pred_answers[0]
			else:
				return INVALID_ANS
		elif type(pred_completion) == dict:
			new_dict = {}
			for key, value in pred_completion.items():
				new_key = str(key)
				new_key = str2num(new_key, rounding, abs_val)
				new_dict[new_key] = value
			return new_dict

	elif dataset_name in ["STRATEGYQA"]:
		answer = bool(pred_completion)
		return answer

	elif dataset_name in ["SPORT"]:
		try:
			answer = bool(int(pred_completion))
		except:
			answer = "[error]"
		return answer

	elif dataset_name in ["SAYCAN"]:
		answer = pred_completion.strip()
		return answer

	else:
		return pred_completion


def line_id_to_step_id(line_id, original_chain, model_name):
	'''Convert a line id to a step id.'''
	output_format = model_name.split("_")[1]
	if output_format == "COT":  # NL chain
		return int(line_id)
	elif output_format == "noNL":  # SL chain
		lines = original_chain.strip().split("\n")
		current_step_id = 0
		for line in lines:
			# get the line id (e.g. '1. ')
			line_id_, line_text = line.split(". ", 1)
			line_id_ = int(line_id_)
			# get the step id
			# if line doesn't start with \s, it's a new step
			if not re.match(r"^\s", line_text):
				current_step_id += 1
			if line_id_ == line_id:
				return current_step_id
	elif output_format == "NL+SL":  # NL+SL chain
		lines = original_chain.strip().split("\n")
		current_step_id = None
		for line in lines:
			# get the line id (e.g. '1. ')
			line_id_ = int(line.split(". ", 1)[0])
			# get the step id (e.g. '# 1.')
			if "#" in line:
				step_id = re.search(r"# (\d+)\.", line).group(1)
				step_id = int(step_id)
				current_step_id = step_id
			if line_id_ == line_id:
				return current_step_id
	else:
		raise NotImplementedError


def remove_line_ids(reasoning_chain):
	'''Remove line ids from reasoning chain.'''
	reasoning_chain = reasoning_chain.strip()
	lines = reasoning_chain.split("\n")
	new_lines = []
	for line in lines:
		pattern = "^\d+\. "
		line = re.sub(pattern, "", line)
		new_lines.append(line)
	reasoning_chain = "\n".join(new_lines)
	return reasoning_chain


if __name__ == "__main__":
	dataset_name = "ASDIV"
	prediction = {"answer": "40 cents"}
	print(extract_pred_answer(dataset_name, prediction))