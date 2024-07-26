# Prompt

This directory contains prompt files for the few-shot learning. Under each dataset directory, there are two folders (`generation/` and `calibration/`), containing the prompts for the generation and calibration tasks, respectively. 

For every prompt with name `{prompt-name}` (e.g., `NL+SL`), there are two files associated with it:
- `{prompt-name}_prompt.txt`: the prompt containing few-shot examples. See `NL+SL_prompt.txt` for an example.
- `{prompt-name}_template.txt`: the template to transform a new input example into the desired format. See `NL+SL_template.txt` for an example.

The above two files will be used in combination to generate the final prompt to query the LM. The `{prompt-name}` here corresponds to the `"prompt_name"` field in a configuration file.