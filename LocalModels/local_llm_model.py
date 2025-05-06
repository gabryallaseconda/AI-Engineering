"""
DOWNLOAD AND EXECUTE LOCALLY AN HUGGINGFACE LLM MODEL

Use config.json to specify the model to download and save locally.
Select your model in the "models_specification" list by choosing the index
on the "model_selected_index" field.

Run this script to download the model from Hugging Face and save it locally.

Import and use load_model function to load the model in you script.
"""




# PSL
import json

# TPL
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch



def _get_configuration(model_selected_index = None, 
                       current_path = "LocalModels/"):
    """
    Read the config.json file and return the configuration parameters.
    """

    # Configuration

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if current_path.endswith("/"):
        current_path = current_path[:-1]
    
    config = json.load(open(f"{current_path}/config.json"))
    
    # Get the model index
    model_selected_index = config["model_selected_index"] if model_selected_index is None else model_selected_index 
    # Get the Hugging Face token
    huggingface_token = config["huggingface_token"] 

    model_owner = config["models_specification"][model_selected_index]["model_owner"]
    model_name = config["models_specification"][model_selected_index]["model_name"]
    local_directory = config["models_specification"][model_selected_index]["local_directory"]

    model_id = f"{model_owner}/{model_name}"
    saving_path = f"{current_path}/{local_directory}"
    
    return device, model_id, saving_path, huggingface_token



def fetch_model(model_selected_index = None,
                current_path = None):
    """
    Fetch the model from Hugging Face and save it locally.
    """

    # Get configuration
    
    device, model_id, saving_path, huggingface_token = _get_configuration(model_selected_index, current_path)

    # Access Hugging Face

    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                            token=huggingface_token)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                token=huggingface_token,
                                                torch_dtype=torch.float16,  # Quantization
                                                device_map=device
    )


    # Save the model locally

    model.save_pretrained(saving_path)

    tokenizer.save_pretrained(saving_path)
    
    
    # End and goodbye
    
    print("Model downloaded and saved successfully.")
    
    
    
def load_model( model_selected_index = None,
                current_path = None,
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50):
    

    # Get configuration
    
    device, _, saving_path, _ = _get_configuration(model_selected_index, current_path)

    # Load saved model
    
    local_tokenizer = AutoTokenizer.from_pretrained(saving_path)

    
    local_model = AutoModelForCausalLM.from_pretrained(saving_path,
                                                       torch_dtype=torch.float16,  # Quantization
                                                       device_map=device
                                                       )
    
    # Initialize the pipeline
    
    text_generator = pipeline(
        "text-generation",
        model=local_model,
        tokenizer=local_tokenizer,
        device_map=device,
        #device=device,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    return text_generator
    
    
if __name__ == "__main__":
    fetch_model()