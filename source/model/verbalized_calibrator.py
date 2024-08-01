import os
cwd = os.getcwd()
if cwd.endswith("source/model"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from keys import API_KEYS, ORGANIZATION_IDS
from dataset.utils import CALIB_STOP_TOKEN_DICT, LM_MAX_TOKEN_DICT, LM_NAME_TO_SHORT_NAME
from calibrate.utils import LING_CONFIDENCE_MAP
import openai
import itertools
import errno
import math
import os
import signal
import functools
import re
import tiktoken

import random
random.seed(42)

# The following are packages/funtions for exponential backoff
# (ref. https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff)
# in order to deal with OpenAI API "rate limit reached" errors
from tenacity import (
	retry,
	stop_after_attempt,
	wait_random_exponential,
)

class TimeoutError(Exception):
	pass

def log_retry(state):
	print(f"Retrying: {state.attempt_number}...")

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
	def decorator(func):
		def _handle_timeout(signum, frame):
			raise TimeoutError(error_message)

		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			signal.signal(signal.SIGALRM, _handle_timeout)
			signal.alarm(seconds)
			try:
				result = func(*args, **kwargs)
			finally:
				signal.alarm(0)
			return result
		return wrapper
	return decorator

class VerbalizedCalibrator():
	'''The VerbalizedCalibrator class.'''
	def __init__(self, config):
		'''Initialize the model with the given configuration.
		@:param config: the configuration object, see source/configuration/configuration.py for details
		'''
		super(VerbalizedCalibrator, self).__init__()

		# dataset parameters
		self.dataset_name = config.dataset_name # name of evaluation dataset
		self.split = config.split # split of the dataset

		self.calib_method = config.calib_method
		assert self.calib_method in ["ptrue", "verb_ling", "verb_percent"]
		self.calib_shots = config.calib_shots

		# core parameters
		self.LM = config.LM
		self.LM_short = LM_NAME_TO_SHORT_NAME[self.LM]
		self.prompt_name = config.prompt_name

		# validate the configuration
		if self.LM_short in ["gpt-3.5-turbo", "gpt4"] and self.calib_method == "ptrue":
			raise NotImplementedError(f"ptrue calibration is not supported for {self.LM_short}, since it does not provide logprobs for top k completions.")
		if self.calib_method != "ptrue" and self.calib_shots > 0:
			raise NotImplementedError(f"Few-shot calibration is only supported for ptrue.")

		# get the stop token for the current dataset
		self.stop_token = CALIB_STOP_TOKEN_DICT[self.dataset_name]
		self.generation_max_tokens = config.max_tokens if config.max_tokens else 10

		# decoding parameters
		self.n_votes = 1 # overwrite config setting
		self.temperature = 0.0 # overwrite config setting
		self.batch_size = 1 # overwrite config setting

		# load the prompt and template
		prompt_path = f"source/prompt/{config.dataset_name}/calibration/{self.calib_method}/{self.prompt_name}_calibration_{self.calib_method}_{self.calib_shots}shot_prompt.txt" # the prompt containing few-shot examples
		template_path = f"source/prompt/{config.dataset_name}/calibration/{self.calib_method}/{self.prompt_name}_calibration_{self.calib_method}_{self.calib_shots}shot_template.txt" # the template to convert a new example

		with open(prompt_path, 'r', encoding='utf-8') as fr:
			self.prompt = fr.read()
		with open(template_path, 'r', encoding='utf-8') as fr:
			self.template = fr.read()

		# openAI tokenizer
		self.tokenizer = tiktoken.encoding_for_model(self.LM)
		self.prompt_n_tokens = 0
		self.template_n_tokens = len(self.tokenizer.encode(self.template))
		self.LM_max_tokens = LM_MAX_TOKEN_DICT[self.LM]

		# load the API keys
		self.api_keys = itertools.cycle(config.api_keys)
		self.org_ids = itertools.cycle(config.org_ids)

	def predict(self, example_dict: dict, completion_only: bool = False):
		'''Predict the answer to an example.
		@:param example_dict (dict): a dict containing the example, in the format of {"question": question, (and other dataset-specific fields)}
		@:param completion_only (bool): whether to only return the completion, but not the answer

		@:return (dict): a output dict, in the format of {"answer": answer,
														  "completion": the final completion,
														  "completions": a list of all completions
												          }
		'''
		# validate reasoning chain format
		if self.dataset_name in ["GSM8K", "CLUTRR"]:
			if self.prompt_name in ["COT", "noNL"]:
				# validate format: reasoning chain starts with a line id (e.g. "1. ")
				assert example_dict["reasoning_chain"].strip().startswith("1. ")
			elif self.prompt_name == "NL+SL":
				# validate format: reasoning chain starts with a hash and a line id (e.g. "# 1. ")
				assert example_dict["reasoning_chain"].strip().startswith("# 1. ")

		self.prompt_n_tokens = len(self.tokenizer.encode(self.prompt))

		# apply the template to the question
		templated_example = self._apply_template(template=self.template, example=example_dict)

		# concatenate the few-shot prompt and the example
		prompt_and_example = f"{self.prompt}\n\n{templated_example}"
		prompt_and_example = prompt_and_example.lstrip()

		# query the LM to get the completions
		n_iters = math.ceil(self.n_votes / self.batch_size) # number of iterations to query the LM
		completions = []
		logprobs = []
		for iter_id in range(n_iters):
			new_completions, new_logprobs = self._query(prompt=prompt_and_example,
								   n=self.batch_size,
								   stop=[self.stop_token],
			                       logprobs = 5 if self.calib_method == "ptrue" else None,
								   max_tokens=self.generation_max_tokens,
								   LM=self.LM,
								   temperature=self.temperature)
			# add completion_prefix to the completions
			new_completions = [completion for completion in new_completions]
			completions += new_completions
			if new_logprobs:
				logprobs += new_logprobs

		output = self.derive_answer_from_completions(completions=completions, logprobs=logprobs)

		return output

	def _apply_template(self, template: str, example: dict):
		'''Apply the template to a new example.
		@:param template (str): the template to be applied to the example.
		@:param example (str): the example to be converted into the template format.

		@:return (str): the example converted into the template format.
		'''
		# for every [{FIELD}] in the template, replace it with the corresponding value of the key "{field}" in the example dict
		example_in_template = template
		field_tokens = {}
		fields_to_replace = {}
		for field in re.findall(r"\[[A-Z_ \-]+?\]", template):
			field_name = field[1:-1]
			field_name = field_name.lower()
			if field_name in example:
				field_in_example = str(example[field_name])
				field_in_example = field_in_example.strip()
				fields_to_replace[field] = field_in_example
				field_n_tokens = len(self.tokenizer.encode(field_in_example))
				field_tokens[field] = field_n_tokens
		max_tokens_allowed = self.LM_max_tokens - self.prompt_n_tokens - self.template_n_tokens - self.generation_max_tokens
		# if total number of tokens in the fields to replace exceeds the max tokens allowed, truncate the longest field
		if sum(field_tokens.values()) > max_tokens_allowed:
			tokens_to_remove = sum(field_tokens.values()) - max_tokens_allowed
			# sort the fields by the number of tokens
			longest_field_name = sorted(field_tokens, key=field_tokens.get, reverse=True)[0]
			longest_field_value = fields_to_replace[longest_field_name]
			# remove the tokens from the longest field
			longest_field_tokens = self.tokenizer.encode(longest_field_value)
			longest_field_tokens_truncated = longest_field_tokens[:-tokens_to_remove]
			longest_field_value_truncated = self.tokenizer.decode(longest_field_tokens_truncated)
			fields_to_replace[longest_field_name] = longest_field_value_truncated
		# replace the fields in the template
		for field in fields_to_replace:
			example_in_template = example_in_template.replace(field, fields_to_replace[field])
		remaining_fields = re.findall(r"\[[A-Z_]+?\]", example_in_template)
		if len(remaining_fields) > 0:
			print(f"ERROR: {len(remaining_fields)} fields are not filled in the template: {remaining_fields}")
		return example_in_template

	def derive_answer_from_completions(self, completions, logprobs=None):
		'''Derive the answer from a list of completions.
		@:param example (dict): the example
		@:param completions (List[str]): the list of completions

		@:return (tuple): answer (type depends on dataset), final_completion (str)
		'''
		if len(completions) == 1:
			completion = completions[0]
			logprob = logprobs[0] if logprobs else None
			try:
				result = self.postprocess_completion(completion=completion, logprob=logprob)  # postprocess the completion
			except Exception as e:
				print(f"Error postprocessing completion: {completion}.\n Error: {e}")
				result = "[invalid]"
			if type(result) in [str, int, float]:
				output = {
					"answer": result,
					"completion": completion,
					"completions": completions
				}
			elif type(result) == dict:
				output = result
				output.update({"completion": completion,"completions": completions})
			else:
				raise TypeError(f"Invalid type for result: {type(result)}")
			return output
		else:
			return NotImplementedError

	def postprocess_completion(self, completion, logprob=None):
		'''Postprocess the completion.
		@:param completion: the completion to be postprocessed

		@:return: the postprocessed answer
		'''
		if self.calib_method == "ptrue":
			logit_dict = {"True": -math.inf, "False": -math.inf}
			for top_token, logit in logprob.items():
				top_token_reformat = top_token.strip(" ()")
				if top_token_reformat not in ["Yes", "No"]:
					continue
				elif top_token_reformat == "Yes":
					logit_dict["True"] = max(logit_dict["True"], logit)
				elif top_token_reformat == "No":
					logit_dict["False"] = max(logit_dict["False"], logit)
			# normalize the logits to get the prob
			prob_True = 10 ** logit_dict["True"] / (10 ** logit_dict["True"] + 10 ** logit_dict["False"])
			confidence = round(prob_True, 3)
			return confidence
		elif self.calib_method == "verb_ling":
			completion = completion.strip()
			completion = completion.split("\n")[0].lower()
			confidence = None
			for expr in LING_CONFIDENCE_MAP:
				if expr in completion:
					confidence = LING_CONFIDENCE_MAP[expr]
					break
			if confidence is None:
				confidence = random.sample(list(LING_CONFIDENCE_MAP.values()), 1)[0]
			confidence = confidence / 100
			return confidence
		elif self.calib_method == "verb_percent":
			completion = completion.strip().split()[0].strip("%")
			try:
				confidence = int(completion)
				confidence = confidence / 100
			except ValueError:
				return "[invalid]"
			return confidence

	@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), after=log_retry)
	def _query(self, prompt, stop, LM, n=1, logprobs=None, temperature=0.0, max_tokens=1024):
		'''Query an OpenAI model.
		@:param prompt (str): the prompt to be fed to the model
		@:param stop (list): the stop tokens
		@:param LM (str): the LM to be queried
		@:param n (int): the number of completions to be returned
		@:param logprobs (int): the number of most likely tokens whose logprobs are to be returned
		@:param temperature (float): the temperature of the model
		@:param max_tokens (int): the maximum number of tokens to be returned

		@:return (dict): the response from the model
		'''
		api_key = next(self.api_keys)
		org_id = next(self.org_ids)
		openai.organization = org_id
		openai.api_key = api_key

		if LM in ["code-davinci-001", "code-davinci-002", "text-davinci-001", "text-davinci-002", "text-davinci-003"]: # models that support "completion"
			response = openai.Completion.create(
				model=LM,
				prompt=prompt,
				temperature=temperature,
				max_tokens=max_tokens,
				logprobs=logprobs,
				n=n,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop
			)
			choices = response["choices"]
			completions = [choice["text"] for choice in choices]
			if logprobs:
				top_logprobs = [choice["logprobs"]["top_logprobs"][0] for choice in choices]
			else:
				top_logprobs = None
		elif self.LM_short in ["gpt-3.5-turbo", "gpt4"]:  # models that support "chat"
			response = openai.ChatCompletion.create(
				model=LM,
				messages=[
					{"role": "user", "content": prompt},
				],
				temperature=temperature,
				n=n,
				frequency_penalty=0,
				presence_penalty=0,
				stop=stop,
				logprobs=True if logprobs else False
			)
			choices = response["choices"]
			completion_objs = [choice.message for choice in choices]
			completions = [completion.content for completion in completion_objs]
			top_logprobs = None
		else:
			raise NotImplementedError(f"Model {LM} is not supported.")
		return completions, top_logprobs


if __name__ == "__main__":
	'''Run a simple test.'''

	dataset_name = "CLUTRR"
	LM = "gpt-3.5-turbo"
	output_format = ["standard", "COT", "LtM", "noNL", "NL+SL"][2]
	n_votes = [1, 40][1]
	if n_votes == 1:
		n_votes_str = ""
	else:
		n_votes_str = f"_n:{n_votes}"

	calib_method = [None, "ptrue", "verb_ling", "verb_percent"][2]
	calib_shots = [0, 8][0]

	config_frn = f"source/configuration/config_files/{LM}_{output_format}{n_votes_str}.json"
	config = Config.from_json_file(config_frn)

	api_key_ids = ['CCB']
	api_keys = [API_KEYS[api_key_id] for api_key_id in api_key_ids]
	org_ids = [ORGANIZATION_IDS[api_key_id] for api_key_id in api_key_ids]

	config.api_keys = api_keys
	config.org_ids = org_ids
	config.dataset_name = dataset_name
	config.calib_method = calib_method
	config.calib_shots = calib_shots

	model = VerbalizedCalibrator(config)

	example = {"id": 7,
	           "answer": "grandson",
	           "entity_0": "[Dan]", "entity_1": "[Gabrielle]",
	           "question": "[Gabrielle] has a grandson who is [Kevin]. [Dan] went to his brother [Kevin]'s Birthday party\nQuestion: How is [Dan] related to [Gabrielle]?",
	           "reasoning_chain": "\n1. How is [Dan] related to [Kevin]?\nDan is Kevin's brother.\n2. How is [Kevin] related to [Gabrielle]?\nKevin is Gabrielle's grandson.\n3. Final answer: How is [Dan] related to [Gabrielle]?\nDan is Gabrielle's grandson.\n\n"}

	output = model.predict(example)
	answer = output["answer"]
	completion = output["completion"]
	print("Answer:", [answer])
	print("Completion:", [completion])
