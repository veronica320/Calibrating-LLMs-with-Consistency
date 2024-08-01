# Configuration

This directory contains code and files related to the model configuration, in order to specify the hyperparameters for each model, such as `"prompt_name"`, `"LM"`, and so on. 

The `configuration.py` file contains the Config class. See the definition of each field in the `init()` funciton.

The `config_files/` directory contains model configuration files. Each configuration file has the name in the format of `{model_name}.json`, e.g., `code002_NL+SL.json`. 
The fields specify hyperparameter values corresponding to those in `source/configuration/configuration.py`.

For example:
- `code002_NL+SL.json`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the greedy decoding strategy.
- `code002_NL+SL_n:40.json`: Faithful CoT prompt (with both NL comments and SL programs) and the code-davinci-002 LM, under the self-consistency decoding strategy (number of generations=40).
