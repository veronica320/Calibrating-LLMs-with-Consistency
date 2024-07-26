LING_CONFIDENCE_MAP = {
	"almost no chance": 5,
	"highly unlikely": 15,
	"unlikely": 25,
	"probably not": 35,
	"about even": 45,
	"better than even": 55,
	"likely": 65,
	"probably": 75,
	"highly likely": 85,
	"almost certain": 95
}

from dataset.utils import extract_gold_answer
from evaluation.evaluate_answer_acc import is_correct
from scipy.stats import pearsonr, ttest_ind
import calibration as cal
import math
from sklearn.metrics import brier_score_loss, auc, precision_recall_curve, roc_auc_score
from collections import Counter

def compute_calibration_error_per_instance(consistency_list, correctness_list):
	# brier score
	brier_per_instance = [brier_score_loss(y_true=[correctness_list[i]], y_prob=[consistency_list[i]]) for i in range(len(correctness_list))]
	return brier_per_instance

def compute_correlation_and_calibration_error(consistency_list, correctness_list, n_bins=10):
	'''Compute correlation and calibration error metrics.'''

	# compute correlation
	correlation, p_value = pearsonr(consistency_list, correctness_list)

	# check if correlation is nan
	if math.isnan(correlation):
		correlation, p_value = None, None

	# compute calibration error metrics

	# expected calibration error
	ece = cal.get_ece(probs=consistency_list, labels=correctness_list, num_bins=n_bins)

	# root-mean-squared calibration error
	rms = cal.get_calibration_error(probs=consistency_list, labels=correctness_list)

	# brier score
	brier_score = brier_score_loss(y_true=correctness_list, y_prob=consistency_list)

	# calibration error for correct and incorrect predictions separately
	# correct
	consistency_list_correct = [consistency_list[i] for i in range(len(correctness_list)) if correctness_list[i] == 1]
	if len(consistency_list_correct) == 0:
		ce_correct = None
	else:
		avg_consistency_correct = sum(consistency_list_correct) / len(consistency_list_correct)
		ce_correct = 1.0 - avg_consistency_correct
	# incorrect
	consistency_list_incorrect = [consistency_list[i] for i in range(len(correctness_list)) if correctness_list[i] == 0]
	if len(consistency_list_incorrect) == 0:
		ce_incorrect = None
	else:
		avg_consistency_incorrect = sum(consistency_list_incorrect) / len(consistency_list_incorrect)
		ce_incorrect = avg_consistency_incorrect

	# Area under the Precision-Recall Curve (AUPRC)
	## with 1 as the positive class
	p_1, r_1, thresh_1 = precision_recall_curve(y_true=correctness_list, probas_pred=consistency_list, pos_label=1)
	auprc_correct = auc(r_1, p_1)
	## with 0 as the positive class
	p_0, r_0, thresh_0 = precision_recall_curve(y_true=correctness_list, probas_pred=[1.0-consistency for consistency in consistency_list], pos_label=0)
	auprc_incorrect = auc(r_0, p_0)

	# Area Under the Receiver Operating Characteristic curve (AUROC)
	try:
		auroc = roc_auc_score(y_true=correctness_list, y_score=consistency_list)
	except ValueError:
		auroc = None

	output = {
		"correlation": correlation,
		"p_value": p_value,
		"ece": ece,
		"rms": rms,
		"brier": brier_score,
		"ce_correct": ce_correct,
		"ce_incorrect": ce_incorrect,
		"auprc_correct": auprc_correct,
		"auprc_incorrect": auprc_incorrect,
		"auroc": auroc
	}
	return output


def compute_correctness(example, prediction, dataset_name, method="majority"):
	pred_answers = prediction["answers"]
	pred_answer = prediction["answer"]
	if not pred_answer:
		pred_answers_strs = [str(a) for a in pred_answers]
		answers_strs_counter = Counter(pred_answers_strs)
		answers_strs_counter_sorted = answers_strs_counter.most_common()
		if answers_strs_counter_sorted[0][0] == "[invalid]" and len(answers_strs_counter_sorted) > 1:
			majority_answer_str = answers_strs_counter_sorted[1][0]
		else:
			majority_answer_str = answers_strs_counter_sorted[0][0]
		majority_answer_id = pred_answers_strs.index(majority_answer_str)
		majority_answer = pred_answers[majority_answer_id]
		pred_answer = majority_answer

	gold_answer = extract_gold_answer(dataset_name, example)
	if method == "majority":
		correct = is_correct(dataset_name, gold_answer, pred_answer)
		correct_binary = 1 if correct else 0
	elif method == "presence":
		correct_binary = 0
		for pred_answer in pred_answers:
			correct = is_correct(dataset_name, gold_answer, pred_answer)
			if correct:
				correct_binary = 1
				break
	else:
		raise NotImplementedError(f"method {method} not implemented")
	return correct_binary

def compute_p_value(brier_score_list1, brier_score_list2):
	# perform t-test
	pvalue = round(ttest_ind(brier_score_list1, brier_score_list2).pvalue, 4)
	return pvalue

if __name__ == "__main__":
	# test
	consistency_lists = [
		[0.1, 0.2, 0.3, 0.4, 0.5],
		[0, 1, 0, 1, 1],
		[1, 1, 1, 1, 1],
		[1, 0, 1, 0, 0],
		[0, 0, 0, 0, 0]
	]
	correctness_list = [0, 1, 0, 1, 1]
	for consistency_list in consistency_lists:
		print(consistency_list)
		# output = compute_correlation_and_calibration_error(consistency_list, correctness_list)
		output = compute_calibration_error_per_instance(consistency_list, correctness_list)
		print(output)
