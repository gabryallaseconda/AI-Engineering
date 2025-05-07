# Interface to Huggingface to use LLMs locally

First, configure which model to use in ```config.json```. Add it to the list called model_specifications, following the convenction. Then write its list index in model_selected_index. The index can be set also as function argument.

The script ```local_llm_model.py``` contains two functions: 
