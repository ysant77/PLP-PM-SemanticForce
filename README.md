Steps to run the code:

1. Make sure to add the model names with apt name in model_names_mapping.json file.
2. Preferably use the smallest models (for ex: gemma-2b in case of chatbot, t5 base for 10k summarization) to save memory.
3. Run the file chatbot_inference.py to get the system running.

Steps to Integrate new functionalities:

1. Refer to the dataset: https://huggingface.co/datasets/yatharth97/10k_reports_gemma, for more information about the user query and assistant
response.

2. For ex if model's response is: document_upload:10K:General Motors Company:2023, then the action here is to store the PDF report uploaded by user in the path: CWD/data/10k/General Motors Company. This folder would then contain the PDF file and the faiss vector database.

3. If the response is: information_extraction:Total Debt:General Motors Company:2023, then the info_extraction_llama.py would be called with appropriate report data and can be used for Question Answering on the document.

4. For the follow up type question, LLM is supposed to answer based on it's previous cached response and the knowledge it has a context.

5. For the summarization question, depending on the content to be summarized (10K/financial report/news), the appropriate model is picked and passed on the TextSummarizer class in summarization_inference.py
