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
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          AutoModelForMaskedLM,
                          pipeline
                          )
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
    model_usage = config["models_specification"][model_selected_index]["usage"]

    model_id = f"{model_owner}/{model_name}"
    saving_path = f"{current_path}/{local_directory}"
    
    allowed_model_usages = ["chat", "embedding"]
    if model_usage not in allowed_model_usages:
        raise ValueError(f"Model type {model_usage} is not supported. Please use one of {allowed_model_types}.")
    
    return device, model_id, model_usage, saving_path, huggingface_token



def fetch_model(model_selected_index = None,
                current_path = None):
    """
    Fetch the model from Hugging Face and save it locally.
    """

    # Get configuration
    
    device, model_id, model_usage, saving_path, huggingface_token = _get_configuration(model_selected_index, current_path)

    # Access Hugging Face

    tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                            token=huggingface_token)

    if model_usage == "chat":
        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                    token=huggingface_token,
                                                    torch_dtype=torch.float16,  # Quantization
                                                    device_map=device
        )
    elif model_usage == "embedding":
        model = AutoModelForMaskedLM.from_pretrained(model_id,
                                                    token=huggingface_token,
                                                    torch_dtype=torch.float16,  # Quantization
                                                    device_map=device
        )
    else:
        raise NotImplementedError(f"Model type {model_usage} is not implemented.")
   
    # Save the model locally
    model.save_pretrained(saving_path,
                          safe_serialization=False if model_selected_index in [2] else True) # otherwise we have problems with bert
    tokenizer.save_pretrained(saving_path)
    
    
    # End and goodbye
    
    print("Model downloaded and saved successfully.")
    
    

def _text_generator_pipeline(local_model,
                             local_tokenizer,
                             device,
                             settings = {}):
    # Initialize the pipeline
    
    text_generator = pipeline(
        "text-generation",
        model=local_model,
        tokenizer=local_tokenizer,
        device_map=device,
        **settings

    )
            


    return text_generator

def _feature_extractor_pipeline(local_model,
                                local_tokenizer,
                                device,
                                settings = {}):
                                
    feature_extraction = pipeline(
        "feature-extraction",
        model=local_model,
        tokenizer=local_tokenizer,
        device_map=device,
        framework="pt",
        return_tensors=True,
        # Importante: configura la pipeline per ottenere gli hidden states
        config={"output_hidden_states": False},
        
        **settings
    )
    
    return feature_extraction




    
def load_model( model_selected_index = None,
                current_path = None,
                settings = {}):
    

    # Get configuration
    
    device, model_id, model_usage, saving_path, _ = _get_configuration(model_selected_index, current_path)
    print(f"Loading model {model_id}...")

    # Load saved model
    
    local_tokenizer = AutoTokenizer.from_pretrained(saving_path)


    if model_usage == "chat":
        
        print("Usage: chat, transformers method: AutoModelForCausalLM, pipeline: text-generation")
        
        local_model = AutoModelForCausalLM.from_pretrained(saving_path,
                                                    torch_dtype=torch.float16,  # Quantization
                                                    device_map=device
                                                    )
        
        
        if settings is None:
            settings = {
                "max_length": 512,
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0.7
            }
        
        return _text_generator_pipeline(local_model,
                                        local_tokenizer,
                                        device,
                                        settings)
        
        
        
    elif model_usage == "embedding":
        
        print("Usage: embedding, transformers method: AutoModelForMaskedLM, pipeline: feature-extraction")
        
        local_model = AutoModelForMaskedLM.from_pretrained(saving_path,
                                                    torch_dtype=torch.float16,  # Quantization
                                                  #  device_map=device
                                                    )
        
        return local_model, local_tokenizer

        
        return _feature_extractor_pipeline(local_model,
                                            local_tokenizer,
                                            device,
                                            settings)
    
    
    else:
        raise NotImplementedError(f"Model type {model_usage} is not implemented.")
    
    
if __name__ == "__main__":
    fetch_model()