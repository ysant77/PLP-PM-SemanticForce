Steps to run the code:

1. Make sure to add the model names with apt name in model_names_mapping.json file.
2. Preferably use the smallest models (for ex: gemma-2b in case of chatbot, t5 base for 10k summarization) to save memory.
    For chatbot use the Llamm2 model (do not change this for now)
3. Run the file chatbot_inference.py to get the system running.

Steps to Integrate new functionalities:

1. Add the corresponding model name and hugging face model name in config/model_names_mapping.json

2. information_extraction works only for PDF based files (10K and financial report)

3. Based on fine tuning of llama2 model, add the if else corresponding to output using subsystems_integration function.

4. Dataset can be referred here: https://huggingface.co/datasets/yatharth97/10k_reports_llama2