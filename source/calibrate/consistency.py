import math

def calculate_consistency_with_main_answer(main_answer, answers):
	# Calculate the percentage of answers that agree with the main answer
	if main_answer == "[invalid]":
		return 0.0

	frequency = {}
	answer_str_to_id = {}
	for answer_id, answer in enumerate(answers):
		answer_str = str(answer)
		answer_str_to_id[answer_str] = answer_id
		if answer_str in frequency.keys():
			frequency[answer_str] += 1
		else:
			frequency[answer_str] = 1

	main_answer_str = str(main_answer)
	if main_answer_str in frequency.keys():
		n_agree = frequency[main_answer_str]
	else:
		n_agree = 0

	consistency = n_agree / len(answers)
	return consistency



def calculate_consistency(answers, method="entropy"):
	# Count the frequency of each unique value
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

	if method == "entropy":
		# Calculate the probabilities
		n_valid_answers = sum(frequency.values())
		probabilities = [f / n_valid_answers for f in frequency.values()]
		# Calculate the entropy
		entropy = - sum(p * math.log2(p) for p in probabilities)
		# normalize entropy to [0, 1]
		if math.log2(len(frequency)) > 0:
			normalized_entropy = entropy / math.log2(len(frequency))
		else:
			normalized_entropy = 0.0
		# deal with numeric instability
		consistency = 1 - normalized_entropy
		if consistency >= 0.0 and consistency <= 1.0:
			pass
		elif consistency < 0.0 and consistency > -1e-6:
			consistency = 0.0
		elif consistency > 1.0 and consistency < 1 + 1e-6:
			consistency = 1.0
		else:
			raise ValueError(f"consistency {consistency} out of range")

	elif method == "agree_percent":
		# Calculate the percentage of predictions that agree with the majority prediction
		agree_percent = max(frequency.values()) / len(answers)
		consistency = agree_percent

	elif method == "fsd": # first-second difference
		# Calculate the first-second difference
		sorted_frequency = sorted(frequency.values(), reverse=True)
		if len(sorted_frequency) > 1: # more than one unique answer
			first_second_difference = sorted_frequency[0] - sorted_frequency[1]
		else: # only one unique answer
			first_second_difference = len(answers)
		# normalize first-second difference to [0, 1]
		normalized_first_second_difference = first_second_difference / len(answers)
		consistency = normalized_first_second_difference

	else:
		raise NotImplementedError(f"method {method} not implemented")

	# check if consistency is in range [0, 1]
	if type(consistency) != float or consistency < 0.0 or consistency > 1.0:
		print(f"consistency {consistency} out of range")
		print(f"answers: {answers}")
		print(f"frequency: {frequency}")
		raise ValueError(f"consistency {consistency} out of range")

	return consistency

if __name__ == "__main__":
	# Example usage
	method = ["entropy", "agree_percent", "fsd"][0]

	vars = ["2", "4", "5", "1", "3", "6", "6"]
	# vars = ["5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5", "2", "35"]
	# vars = [0, 0, 0, 0, '[invalid]', '[invalid]', 0, '[invalid]', 0, '[invalid]', 0, '[invalid]', 23, 23, '[invalid]', 23, '[invalid]', 23, 23, '[invalid]', '[invalid]', 0, 23, 23, 23, 0, 0, 0, '[invalid]', 23, 0, 0, '[invalid]', 23, 23, 23, 0, 23, 23, 0]
	print(len(vars))

	consistency = calculate_consistency(vars, method=method)
	print(consistency)