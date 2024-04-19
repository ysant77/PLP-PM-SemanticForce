from utils import extract_model_responses
from constants import system_prompt
def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

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