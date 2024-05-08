import os
from transformers import BitsAndBytesConfig
import torch

curr_dir = os.getcwd()
model_names_json_mapping_path = f'{curr_dir}/Backend/config/model_names_mapping.json'
secrets_path = f'{curr_dir}/Backend/config/secrets.ini'
bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

valid_constructs = ["document_upload", "information_extraction", "summarize", "news_fetch_and_summarize", "news_sentiment", "rag"]

dummy_pdf_path = f'{curr_dir}/Backend/PDF_files/Apple.pdf'

system_prompt = """
As a highly intelligent assistant and successor of google gemma model, your primary goal is to provide accurate, relevant, and context-aware responses to
user queries based on the provided information. Ensure your answers are factual, free from bias, and avoid promoting violence, hate speech, or any form
of discrimination. Focus on assisting the user effectively and safely. Also do not include user's query in response again
"""
llama2_system_prompt = ("<<SYS>> As a highly intelligent assistant, your primary goal is to provide accurate, "
                 "relevant, and context-aware responses to user queries based on the provided information. "
                 "Ensure your answers are factual, free from bias, and avoid promoting violence, hate speech, "
                 "or any form of discrimination. Focus on assisting the user effectively and safely. <</SYS>>")


data_dir = f'{curr_dir}/Backend/data'

local_llm_model_path = f'{curr_dir}/Backend/models/llama-2-7b-chat.ggmlv3.q2_K.bin'