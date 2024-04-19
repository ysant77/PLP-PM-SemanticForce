from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import json
from constants import model_names_json_mapping_path, secrets_path, bnbConfig, system_prompt
from utils import extract_model_responses

model_names_map = json.load(open(model_names_json_mapping_path))
modelName = model_names_map['gemma2B']
model = AutoModelForCausalLM.from_pretrained(
        modelName,
        quantization_config=bnbConfig
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(modelName)
tokenizer.padding_side='left'

query = "I have uploaded the 10K report for Apple for the year 2023"
    
def get_completion(query: str, model, tokenizer) -> str:
  device = torch.device("cpu")

  prompt_template = """
  <start_of_turn>user
  {system_prompt}
  {query}
  <end_of_turn>\n<start_of_turn>model


  """
  prompt = prompt_template.format(system_prompt=system_prompt, query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, early_stopping=True, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
  
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  model_response = decoded.split("<start_of_turn>model\n\n")[-1].strip()

  return model_response

def post_processing_model_output(model_output):
    
    responses = extract_model_responses(model_output)
    output = [res.strip() for res in responses if res.startswith("model")]
    output = [ele.replace("model", "").strip() for ele in output]

    response = output[0]
    return response

model_output = get_completion(query, model, tokenizer)
processed_output = post_processing_model_output(model_output)

