'''Evaluate the answer accuracy of a model output file.'''

import os
if os.getcwd().endswith("evaluation"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data, extract_gold_answer, extract_pred_answer
import argparse
import regex as re
import pandas as pd

def is_correct(dataset_name, gold_answers, pred_answer):
	'''Check if a predicted answer is correct.
	:param dataset_name (str): The name of the dataset.
	:param gold_answers: The gold answer(s).
	:param pred_answer: The predicted answer.

	:return: Whether the prediction is correct (True) or not (False).
	'''

	# saycan has multiple correct plans, so we need to check if the predicted plan is in the list of correct plans
	if dataset_name == "SAYCAN":
		assert type(gold_answers) == list
		assert type(pred_answer) == str
		if pred_answer in ["[error]", "[invalid]"]:
			return False
		else:
			pred_answer = pred_answer.replace("\\n", "\n")
			pred_plan_list = []
			step_count = 0
			steps = re.split(r", |\n", pred_answer.strip())
			for step in steps:
				step_cols = step.split(". ")
				if len(step_cols) != 2:
					return "[invalid]"
				step_action = step_cols[1]
				if "find(initial)" in step_action:
					continue
				step_count += 1
				new_step = f"{step_count}. {step_action}"
				pred_plan_list.append(new_step)
			for gold_answer in gold_answers:
				gold_plan_list = gold_answer.strip().split("\n")
				if pred_plan_list == gold_plan_list:
					return True
		return False

	else:	# all other datasets have a single correct answer
		gold_answer = gold_answers
		return pred_answer == gold_answer

def custom_accuracy_score(gold_answers, pred_answers, dataset_name):
	'''Compute the accuracy score.
	:param gold_answers (list): The gold answers.
	:param pred_answers (list): The predicted answers.

	:return: The accuracy score.
	'''
	correct_count = 0
	total_count = 0
	for gold_answer, pred_answer in zip(gold_answers, pred_answers):
		if is_correct(dataset_name, gold_answer, pred_answer):
			correct_count += 1
		total_count += 1
	if total_count == 0:
		acc = None
	else:
		acc = correct_count / total_count
	return acc

def evaluate_acc(dataset,
                 predictions,
                 dataset_name,
                 debug=False,
                 ):
	'''Evaluate the answer accuracy of a model output file.
	:param dataset (list): The dataset.
	:param predictions (list): The predictions.
	:param dataset_name (str): The name of the dataset.
	:param debug (bool): Whether to only evaluate the first 10 examples.

	:return score_dict (dict): A dictionary of evaluation scores, in the format of
	{
		"overall": {"acc": score}, # overall accuracy
	}
	'''
	answers_dict = {
		"overall": {"gold_answers": [], "pred_answers": []}, # overall
	}

	total_count = 0

	for example, prediction in zip(dataset, predictions):
		gold_id = int(example["id"])
		if prediction == {}:
			continue
		pred_id = int(prediction["id"])

		assert gold_id == pred_id

		# extract the gold answer and the predicted answer
		try:
			gold_answer = extract_gold_answer(dataset_name, example)
			pred_answer = extract_pred_answer(dataset_name, prediction)
		except Exception as e:
			print("Error in extracting answers: ", e)
			print("Example: ", gold_id)
			print("Question: ", example["question"])
			print("Gold answer: ", example["answer"])
			print("Pred answer: ", prediction["answer"])
			print("Completion: ", prediction["completion"])
			print("\n")
			exit(-1)

		answers_dict["overall"]["gold_answers"].append(gold_answer)
		answers_dict["overall"]["pred_answers"].append(pred_answer)

		total_count += 1
		if debug and total_count >= 10:
			break

	## compute overall acc
	acc = custom_accuracy_score(answers_dict["overall"]["gold_answers"], answers_dict["overall"]["pred_answers"], dataset_name)
	acc = round(acc * 100, 1) if acc is not None else None
	score_dict = {"overall": {"acc": acc}}

	return score_dict

if __name__ == "__main__":
	Parser = argparse.ArgumentParser()
	Parser.add_argument("--dataset_name", help="The name of the dataset.", required=True)
	Parser.add_argument("--split", help="The split of the dataset.", required=True)
	Parser.add_argument("--preds_fn", help="The name of the predictions file.", required=True)
	Parser.add_argument("--LM", help="The name of the LM.", required=True, choices=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-instruct", "olmo-7B", "olmo-7B-instruct", "olmo-7B-instruct-rlhf"])
	Parser.add_argument("--output_format", help="The format of the output.", choices=["standard", "COT", "noNL", "NL+SL", "LtM"], required=True)
	Parser.add_argument("--n_vote", help="The number of votes", type=int, default=1)
	Parser.add_argument("--non_empty_only", help="If true, only evaluate on non-empty answers.", action="store_true")
	Parser.add_argument("--valid_only", help="If true, only evaluate on valid answers.", action="store_true")
	Parser.add_argument("--debug", help="If true, only run on the first 10 examples.", action="store_true")

	# Example usage:
	# python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM gpt4 --output_format COT --n_vote 40 --preds_fn predictions.jsonl

	args = Parser.parse_args()
	dataset_name = args.dataset_name
	split = args.split
	debug = args.debug
	non_empty_only = args.non_empty_only
	valid_only = args.valid_only
	LM = args.LM
	output_format = args.output_format
	n_vote = args.n_vote
	preds_fn = args.preds_fn

	if n_vote > 1:
		n_str = f"_n:{n_vote}"
	else:
		n_str = ""

	# load the dataset and predictions
	dataset_frn = f"data/{dataset_name}/{split}.jsonl"
	output_dir = f"output_dir/{dataset_name}/{split}/{LM}_{output_format}{n_str}"
	pred_frn = f"{output_dir}/{preds_fn}"

	dataset = load_data(dataset_frn)
	predictions = load_data(pred_frn)
	dataset_wid = load_data(dataset_frn, key="id")
	predictions_wid = load_data(pred_frn, key="id")

	if not debug:
		try:
			assert len(dataset) == len(predictions)
		except:
			print(f"Dataset {dataset_name} has {len(dataset)} examples, but predictions file {pred_frn} has {len(predictions)} predictions. Skipping...")
			exit(-1)

	score_dict = evaluate_acc(dataset=dataset,
	                   predictions=predictions,
	                   dataset_name=dataset_name,
	                   debug=debug
					   )

	print(f"\nDataset: {dataset_name}\nSplit: {split}\nModel: {LM}_{output_format}{n_str}\nPreds frn:{pred_frn}\n")
	if debug:
		print("--debug")

	score_dir =  f"score_dir/answer_accuracy/{dataset_name}/{split}/{LM}_{output_format}{n_str}"
	if not os.path.exists(score_dir):
		os.makedirs(score_dir)

	preds_fn_raw = preds_fn.split("/")[-1]
	# save as a csv file
	fwn = f"{score_dir}/generation_scores{'_debug' if debug else ''}_{preds_fn_raw}.csv"
	acc_df = pd.DataFrame.from_dict(score_dict, orient="index")
	acc_df.to_csv(fwn)
	print(acc_df)