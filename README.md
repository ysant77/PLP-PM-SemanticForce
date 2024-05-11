
# SystemCode Structure

## Directory Layout
```
SystemCode/
│
├── requirements.txt
│
├── ModelTraining/
│   ├── .10K_reports/
│   ├── Fine_tune_Llama_2_in_Google_Colab_Final.ipynb
│   ├── News_data_training_Gemini.ipynb
│   ├── News_summarization_fine_tuning_T5.ipynb
│   ├── Annual_report_summarization_Fine_tuning_T5.ipynb
│   └── document_QA_news.ipynb
│
├── Frontend/
│   └── app.py
│
└── Backend/
    ├── .config/
    ├── .data/
    ├── .models/
    ├── .PDF_files/
    ├── .RAG/
    ├── .results/
    ├── chatbot_inference.py
    ├── constants.py
    ├── gemma_inference.py
    ├── info_extraction_llama.py
    ├── news_extraction_sentiment.py
    ├── persistence.py
    ├── README.md (obsolete)
    ├── summarization_inference_10k.py
    ├── summarization_inference_annual.py
    └── utils.py

Dataset
│
├── BofA Annual Report.pdf
└── Apple 10K Report.pdf

Presentation Video
│
├── PLP Technical Slides.pdf
└── readme.md

Project Report
│
├── Grp 1 PLP Porject Report.pdf
└── readme.md

Test_Script.txt
README.md
```

## Brief Description
The `app.py` in the Frontend directory serves as both the main script and the webpage design. User prompts to the chatbot trigger model inferencing like the various T5 models for different tasks.

The Dataset folder contains financial report to test on the model and the Test_Script file contains user prompts to send to the chatbot to try.

## Instructions on How to Run the Code

1. Navigate to the SystemCode directory:
   ```bash
   cd SystemCode/
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Gemini API token in `Backend/RAG/.env`:
   - [Gemini](https://aistudio.google.com/app/u/2/apikey)

4. Set up HuggingFace token in `Backend/config/secrets.ini`:
   - [HuggingFace](https://huggingface.co/)

5. Download the models' weights from [Google Drive](https://drive.google.com/file/d/1Pjg4JFzGy0g_pbf4-t1m823epq9R_CZ1/view?usp=drive_link), unzip and upload to `Backend/models/`.

6. Make sure to add the model names with apt name in Backend/config/model_names_mapping.json file.

7. Preferably use the smallest models (for ex: gemma-2b in case of chatbot, t5 base for 10k summarization) to save memory.
    For chatbot use the Llamm2 model (do not change this for now)

8. Run the application:
   ```bash
   streamlit run Frontend/app.py
   ```


## Steps to Integrate new functionalities:

1. Add the corresponding model name and hugging face model name in Backend/config/model_names_mapping.json

2. information_extraction works only for PDF based files (10K and Annual report)

3. Based on fine tuning of llama2 model, add the if else corresponding to output using subsystems_integration function.

4. Dataset can be referred here: https://huggingface.co/datasets/yatharth97/PLP_llama2_v1

## Reference code for dataset generation and model training

- Model training code is given in the SystemCode/ModelTraining folder.

## System Requirements

- **RAM**: Minimum of 32GB required to run the Llama2-7b and T5 models

## Additional Resources

- **Llama2 Model**: Check out the Llama2-7b model [here](https://huggingface.co/gmh98/llama-2-7b-chat-yatharth-v4).
- **Dataset**: The dataset used to fine-tune the Llama2-7b model is available [here](https://huggingface.co/datasets/yatharth97/PLP_llama2_v1).
- **T5 Base Model For 10K Report Summarization**: Check out the T5 model [here](https://huggingface.co/yatharth97/T5-base-10K-summarization).
- **T5 Base Model For Annual Report Summarization**: Check out the T5 model [here](https://huggingface.co/Kgr20/AnnualSummarizer).
- **T5 Base Model For News Sentiment Analysis**: Check out the T5 model [here](https://huggingface.co/yatharth97/T5-base-news-summarization).


