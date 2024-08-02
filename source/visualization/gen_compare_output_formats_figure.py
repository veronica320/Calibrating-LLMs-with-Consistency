'''Generate the table to compare calibration performance across baselines.'''
import os
cwd = os.getcwd()
if cwd.endswith("source/visualization"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_names", nargs="+", default=["ASDIV", "CLUTRR", "GSM8K", "DATE", "MULTIARITH", "SAYCAN", "SPORT", "STRATEGYQA", "SVAMP"])
	parser.add_argument("--LMs", nargs="+", default=["code002", "gpt-3.5-turbo", "gpt4", "llama-7B", "llama-13B", "llama-70B", "mistral-7B", "mistral-7B-it", "olmo-7B", "olmo-7B-it", "olmo-7B-it-rl"])
	parser.add_argument("--output_formats", nargs="+", default=["standard", "COT", "LtM", "noNL", "NL+SL"])
	parser.add_argument("--split", type=str, default="test", choices=["test", "dev_100"])
	parser.add_argument("--n_vote", type=int, default=40)
	parser.add_argument("--consistency_metrics", nargs="+", default=["entropy", "agree_percent", "fsd"])
	parser.add_argument("--n_bins", type=int, default=10)
	parser.add_argument("--debug", action="store_true")
	parser.add_argument("--vals", nargs="+", default=["ece", "brier"], help="The calibration error metrics to compute. Options: correlation, p_value, ece, rms, brier, auroc, ce_correct, ce_incorrect, auprc_correct, auprc_incorrect")
	args = parser.parse_args()

	dataset_names = args.dataset_names
	LMs = args.LMs
	output_formats = args.output_formats
	split = args.split
	n_vote = args.n_vote
	debug = args.debug
	n_bins = args.n_bins
	vals = args.vals
	consistency_metrics = args.consistency_metrics
	confidence_metrics = consistency_metrics
	correctness_metric = "majority"

	if n_vote > 1:
		n_str = f"_n:{n_vote}"
	else:
		n_str = ""

	all_results = {
		val: {
			LM:{
				output_format: None for output_format in output_formats
			} for LM in LMs
		} for val in vals
	}
	for val in vals:
		for LM in LMs:
			LM_results = {
				output_format: {
					confidence_metric: None for confidence_metric in confidence_metrics
				} for output_format in output_formats
			}
			for output_format in output_formats:
				config_name = f"{LM}_{output_format}{n_str}"
				output_dir = f"score_dir/calibration_error/{LM}"
				frn = f"{output_dir}/calerror_across_methods_{output_format}_{correctness_metric}.csv"
				results = pd.read_csv(frn, index_col=[0, 1]).to_dict()
				# convert to dict
				for confidence_metric in confidence_metrics:
					LM_results[output_format][confidence_metric] = results[confidence_metric][(val, "average")]
				# compute average across confidence metrics
				all_results[val][LM][output_format] = np.mean(list(LM_results[output_format].values()))
				all_results[val][LM][output_format] = "{:.3f}".format(all_results[val][LM][output_format])

		# save results
		output_dir = f"score_dir/calibration_error/all_LMs"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		df = pd.DataFrame(all_results[val])
		# switch rows and columns
		df = df.transpose()
		# convert to Excel
		excel_path = f"{output_dir}/calerror_across_output_formats_{val}.xlsx"
		df.to_excel(excel_path)

		# Attempt to read the converted Excel file
		data_converted = pd.read_excel(excel_path)
		data_converted.rename(columns={data_converted.columns[0]: 'Model'}, inplace=True)
		data_melted = data_converted.melt(id_vars='Model', var_name='Method', value_name='Score')

		# Define a list of hatch patterns to use for the bars
		hatch_patterns = ['', '/', '\\', '-', '.']

		# Create the bar plot again with the updated legend and colorblind-friendly patterns
		plt.figure(figsize=(12, 3))

		palette = ["Set2", "tab10", "deep"][2]
		# bar_plot = sns.barplot(data=data_melted, x='Model', y='Score', hue='Method', palette=tab10_palette)
		bar_plot = sns.barplot(data=data_melted, x='Model', y='Score', hue='Method', palette=palette)

		# Map the 'Method' to a hatch pattern, with white edge color
		method_to_hatch = {method: hatch for method, hatch in zip(data_melted['Method'].unique(), hatch_patterns)}
		# Apply hatch patterns by 'Method' category
		for bar, method in zip(bar_plot.patches, data_melted['Method']):
			bar.set_hatch(method_to_hatch[method])
			bar.set_edgecolor('white')  # This will make the hatch color white

		# Set the title and labels
		ylabel_name = "Brier Score" if val == "brier" else "ECE"
		plt.ylabel(ylabel_name, fontsize=14)
		# omit the x-axis label
		plt.xlabel('')
		# set the x-axis category name font size
		plt.xticks(fontsize=10)

		# Adjust the legend to be on top of the figure and not block any bars

		legend_handles = [mpatches.Patch(facecolor=col, hatch=hatch, label=method, edgecolor='white')
		                  for method, hatch, col in zip(data_melted['Method'].unique(), hatch_patterns, sns.color_palette(palette))]

		# Add the legend to the top of the figure, as a horizontal legend
		plt.legend(handles=legend_handles,  loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=12)

		# Adjust the layout
		plt.tight_layout()

		# Save the updated plot to a file
		plot_path = f"{output_dir}/calerror_across_output_formats_{val}_{palette}.pdf"
		plt.savefig(plot_path, bbox_inches='tight', dpi=300, format="pdf")
		plt.close()

