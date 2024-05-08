# !pip install PyPDF2 transformers
# !pip install torch -U
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Function to read and extract text from each page of the PDF
def extract_text_by_page(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            yield text


# Function to summarize a piece of text using the model
def summarize_text(model, tokenizer, text):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Main pipeline function
def TextSummarizer_annual(pdf_path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    # Extract text from PDF
    pages_text = extract_text_by_page(pdf_path)

    # Summarize each page
    summaries = [summarize_text(model, tokenizer, page_text) for page_text in pages_text]

    # Concatenate all summaries
    final_summary = "\n".join(summaries)

    return final_summary


# # Use the pipeline function
# pdf_path = '/content/drive/MyDrive/BofA Annual Report.pdf'
# model_name = 'kgr20/AnnualSummarizer'
# final_summary = pdf_summary_pipeline(pdf_path, model_name)