import re
import os
from constants import data_dir
import pandas as pd

def extract_model_responses(text: str) -> list:
    # Pattern to match everything after "model" till it potentially hits another "user" or end of string
    pattern = r"model(.*?)(?=user|$)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    # Clean and return the matches
    return [match.strip() for match in matches if match.strip()]

def generate_path(args):
    report_dir = args[1]
    company_dir = args[2]
    year_dir = args[3]
    # dir_path = f'{data_dir}/{report_dir}/{company_dir}/{year_dir}'
    dir_path = f'{data_dir}/{company_dir}/{year_dir}'
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def fetch_qa_llm_response(question, model):
    output = model({'query': question})
    response = output["result"]
    return response

def summarize_df_10k(df_path, summarizer, company_name):
    df_path_arr = df_path.split("/")
    data_dir = "/".join(df_path_arr[:-1])

    df = pd.read_csv(df_path)
    # Apply the summarization to each section in the DataFrame
    df['Section Text'] = 'summarize: ' + df['Section Text']
    df['Summary'] = df['Section Text'].apply(lambda text: summarizer.summarize(text))
    output_df_path = f'{data_dir}/10k_{company_name}_summary.csv'
    df.to_csv(output_df_path, index=None)
    return df, output_df_path


def summarize_df_news(df_path, summarizer, company_name):
    df_path_arr = df_path.split("/")
    data_dir = "/".join(df_path_arr[:-1])

    df = pd.read_csv(df_path)
    # Apply the summarization to each section in the DataFrame
    df['article_text'] = 'summarize: ' + df['article_text']
    df['Summary'] = df['article_text'].apply(lambda text: summarizer.summarize(text))
    output_df_path = f'{data_dir}/news_{company_name}_summary.csv'
    df.to_csv(output_df_path, index=None)
    return df, output_df_path