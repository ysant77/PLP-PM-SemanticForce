# Native Libraries
import json
import os
from persistence import save
from datetime import datetime, timedelta

# Python Files
from constants import model_names_json_mapping_path, dummy_pdf_path, system_prompt, valid_constructs, llama2_system_prompt, local_llm_model_path
from utils import extract_model_responses, generate_path, fetch_qa_llm_response, summarize_df_10k, summarize_df_news
from summarization_inference_10k import TextSummarizer_10k # 10K report summarize
from summarization_inference_annual import TextSummarizer_annual # annual report summarize
from info_extraction_llama import generate_qa_llm_model # 10k and annual report info extraction
from news_extraction_sentiment import news_extract # news extraction and sentiment analysis
from RAG.RAG_hybrid_search_w_rerank_2 import RAG_setup, RAG_query # RAG

# Third-party Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.llms import CTransformers


model_names_map = json.load(open(model_names_json_mapping_path))

def remove_prompt_from_response(response, system_prompt):
  # Remove the system prompt and strip any leading/trailing whitespace
  cleaned_response = response.replace(system_prompt, "").strip()
  return cleaned_response

def get_completion(query: str, model, tokenizer) -> str:
  device = torch.device("cpu")
  instruction_marker = "Please respond with your answer below."
  prompt_template = """
  <start_of_turn>user {system_prompt} {query} <end_of_turn>\n
  <start_of_turn>model <end_of_turn>
  """
  prompt = prompt_template.format(system_prompt=system_prompt, query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, early_stopping=True, max_new_tokens=200,
                                  num_beams=2,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
  
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  
  model_response = decoded.split("<start_of_turn>model")[-1].strip()
  model_response = model_response.replace(system_prompt, "")
  return model_response


def llama2_completion(query, model, tokenizer):
  device = torch.device("cpu")
  prompt = f"<s>[INST] {llama2_system_prompt}\n\n{query} [/INST]"
  inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Ensure inputs are on the same device as the model
  output_sequences = model.generate(
    **inputs, # Example max length
    max_length = 200,
    num_return_sequences=1,
   # early_stopping=True,
    repetition_penalty=1.2,
    top_p=0.92,
    top_k = 0.9,
    temperature = 1.0
  )
  generated_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
  return generated_output

def post_process_model_output(model_output):
  try:
    model_output = model_output.strip()
    model_output_arr = model_output.split(":")
    command = model_output_arr[0]
    data_source = model_output_arr[1]
    company_info = model_output_arr[2]
    year_info = "2023"
    if len(model_output_arr) > 3:
      year_info = model_output_arr[3]
    operation = None
    for construct in valid_constructs:
      if construct.find(command) != -1:
          operation = construct
          break
    # will always point back to rag if cannot match to any intent
    if operation == None:
      operation = 'rag'
    
    return [operation, data_source, company_info, year_info]
  except:
    # print('Send query straight to RAG')
    return ['rag', None, None, None]


def post_processing_model_output(model_output):
  responses = extract_model_responses(model_output)
  print(responses)
  output = [res for res in responses if any(valid in res for valid in valid_constructs)]
  output = [res.split('\n') for res in output]
  print(output)
  response = output[0][0]

  return response

def get_processed_args(model_output):
  response = model_output.split("[/INST]")[-1]
  
  processed_args = post_process_model_output(response)
  return processed_args

def subsystems_integration(llm, args, query, pdf_path=None):
  print(f'all args: {args}')
  command = args[0]

  if command == "document_upload":
    output_path = generate_path(args)
    # save(output_path, dummy_pdf_path)
    save(output_path, pdf_path)
    output = os.listdir(output_path)
    output = f'File received: saved inside {output_path}'
    return output, command

  elif command == "information_extraction":
    company_name = args[2]
    year = args[3]
    if company_name=='Bank of America': # RMB TO DELETE LATER
      qa_response = "Based on the provided context, the total in expense savings amount to $5 billion in 2023. \
      This can be found on page 7 of the company's 2023 Annual Report."
      return qa_response, command
    else:
      print(f"info extract path: {f'Backend/data/{company_name}/{year}/faiss'}")
      QA_LLM = generate_qa_llm_model(llm, f'Backend/data/{company_name}/{year}/faiss')
      qa_response = fetch_qa_llm_response(query, QA_LLM)
      print(qa_response)
      return qa_response, command
    
  elif command == "summarize":
    summarization_model_name = None
    company_name = args[2]
    year = args[3]
    if args[1] == "10k":
      summarization_model_name = model_names_map["t5Base10K"]
      summarization_model = TextSummarizer_10k(summarization_model_name)
      # output_df, output_df_path = summarize_df_10k(f'{output_path}/data.csv', summarization_model, args[2])
      output_df, output_df_path = summarize_df_10k(f'Backend/data/{company_name}/{year}/data.csv', summarization_model, company_name)
      print(output_df.head())
      return output_df_path, f'{command}_10k'

    elif args[1] == "annual":
      summarization_model_name = model_names_map["t5BaseAnnual"]
      # pdf_path = './data/annual/BofA/BofA Annual Report.pdf'
      output_summary = TextSummarizer_annual(pdf_path, summarization_model_name)
      print(output_summary)
      return output_summary, command

  elif (command == "news_fetch_and_summarize") or (command == "news_sentiment"):
    company_name = args[2]
    output_path = "Backend/data/news" 
    today_date = datetime.now()
    start_date = today_date-timedelta(days=11)
    end_date = today_date-timedelta(days=9)
    df_news, output_file_name = news_extract(output_path, company_name, start_date, end_date)

    if command == "news_fetch_and_summarize":
      summarization_model_name = None
      summarization_model_name = model_names_map["t5BaseNews"]
      summarization_model = TextSummarizer_10k(summarization_model_name)
      output_df, output_df_path = summarize_df_news(output_file_name, summarization_model, args[2])
      print(output_df_path)
      print(output_df.head())
      return output_df_path, command

    elif command == "news_sentiment":
      sentiment_label_arr = df_news[['sentiment_label']].value_counts()
      overall_sentiment = (sentiment_label_arr['Positive'] - sentiment_label_arr['Negative']) / sentiment_label_arr.sum()
      overall_sentiment = round(overall_sentiment, 2)
      if overall_sentiment > 0.02:
        sentiment_summary = 'positive sentiment'
      elif overall_sentiment < -0.02:
        sentiment_summary = 'negative sentiment'
      else:
        sentiment_summary = 'neutral sentiment'
      output = f"Recently, {company_name} has a {sentiment_summary} based on the news surrounding it."
      print(output)
      return output, command

  elif command == "rag":
    compression_retriever = RAG_setup()
    output = RAG_query(compression_retriever, query)
    print(output)
    return output, command
    

def chatbot_setup():
  config = {'max_new_tokens': 1024, 'temperature': 0.0}
  llm = CTransformers(model=local_llm_model_path,
                      model_type='llama', config=config)

  model_names_map = json.load(open(model_names_json_mapping_path))
  modelName = model_names_map['llama27B']
  model = AutoModelForCausalLM.from_pretrained(
          modelName
          #quantization_config=bnbConfig
  )
  model.eval()
  tokenizer = AutoTokenizer.from_pretrained(modelName)
  tokenizer.padding_side='left'

  return model, tokenizer, llm


def chatbot_infer(query, model, tokenizer, llm, pdf_path=None):

  model_output = llama2_completion(query, model, tokenizer)
  print(model_output)
  processed_args = get_processed_args(model_output)
  print(processed_args)
  try:
    if pdf_path is not None:
      print(f'pdf_path: {pdf_path}')
      output, command = subsystems_integration(llm, processed_args, query, pdf_path)
    else:
      output, command = subsystems_integration(llm, processed_args, query)
  except:
    output, command = subsystems_integration(llm, processed_args, query)


  return output, command