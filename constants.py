import os
from transformers import BitsAndBytesConfig
import torch

model_names_json_mapping_path = f'{os.getcwd()}/config/model_names_mapping.json'
secrets_path = f'{os.getcwd()}/config/secrets.ini'
bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

system_prompt = """
As a highly intelligent assistant and successor of google gemma model, your primary goal is to provide accurate, relevant, and context-aware responses to
user queries based on the provided information. Ensure your answers are factual, free from bias, and avoid promoting violence, hate speech, or any form
of discrimination. Focus on assisting the user effectively and safely. Also do not include user's query in response again
"""