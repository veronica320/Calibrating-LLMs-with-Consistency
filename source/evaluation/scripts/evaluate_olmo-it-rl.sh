cd ../

############################ ASDIV #############################


python evaluate_answer_acc.py --dataset_name ASDIV --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name ASDIV --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name ASDIV --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name ASDIV --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name ASDIV --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ CLUTRR #############################


python evaluate_answer_acc.py --dataset_name CLUTRR --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name CLUTRR --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name CLUTRR --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name CLUTRR --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name CLUTRR --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ DATE #############################


python evaluate_answer_acc.py --dataset_name DATE --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name DATE --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name DATE --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name DATE --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name DATE --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ GSM8K #############################


python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name GSM8K --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ MULTIARITH #############################


python evaluate_answer_acc.py --dataset_name MULTIARITH --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name MULTIARITH --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name MULTIARITH --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name MULTIARITH --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name MULTIARITH --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ SAYCAN #############################


python evaluate_answer_acc.py --dataset_name SAYCAN --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SAYCAN --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SAYCAN --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SAYCAN --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SAYCAN --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"


############################ SPORT #############################


python evaluate_answer_acc.py --dataset_name SPORT --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SPORT --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SPORT --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SPORT --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SPORT --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ STRATEGYQA #############################


python evaluate_answer_acc.py --dataset_name STRATEGYQA --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name STRATEGYQA --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name STRATEGYQA --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name STRATEGYQA --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name STRATEGYQA --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"

############################ SVAMP #############################


python evaluate_answer_acc.py --dataset_name SVAMP --split test --LM olmo-7B-it-rl --output_format standard --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SVAMP --split test --LM olmo-7B-it-rl --output_format COT --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SVAMP --split test --LM olmo-7B-it-rl --output_format noNL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SVAMP --split test --LM olmo-7B-it-rl --output_format NL+SL --n_vote 40 --preds_fn "predictions.jsonl"
python evaluate_answer_acc.py --dataset_name SVAMP --split test --LM olmo-7B-it-rl --output_format LtM --n_vote 40 --preds_fn "predictions.jsonl"