#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain')


# In[2]:


from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from IPython.display import display, HTML
import json
import time
import os
import pathlib


# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import pandas as pd

# Load the Excel file
excel_file_path = '/content/drive/MyDrive/Plp news/news_summary_trainingv2.xlsx'  # Update with your file path
df = pd.read_excel(excel_file_path)

# Read the original article data from the first column and first row (assuming your data starts from A1)
original_article = df.iloc[0, 0]


# Split the original article into smaller chunks
chunk_size = 500
texts = [original_article[i:i+chunk_size] for i in range(0, len(original_article), chunk_size)]

# Now you have a list of smaller chunks stored in the 'texts' variable
print(texts)




# In[5]:


# data_dir = f'{os.getcwd()}/PDF_files/data'


# In[6]:


# loader = DirectoryLoader(path=data_dir, glob="*.pdf", loader_cls=PyPDFLoader)


# In[7]:


# # documents = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                           chunk_overlap=50)
# texts = splitter.split_documents(documents)


# In[8]:


get_ipython().system('pip install sentence-transformers')


# In[9]:


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})


# In[10]:


get_ipython().system('pip install faiss-cpu')
get_ipython().system('pip install faiss-gpu')


# In[11]:


# db = FAISS.from_documents(texts, embeddings)
# db.save_local("faiss")

# Initialize the vector store from texts
db = FAISS.from_texts(texts, embeddings)
db.save_local("faiss")


# In[12]:


get_ipython().system('pip install ctransformers')


# In[13]:


template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".
    Context: {context}
    Question: {question}
    Answer:
    """

# load the language model
config = {'max_new_tokens': 256, 'temperature': 0.0}
llm = CTransformers(model='/content/drive/MyDrive/llama-2-7b-chat.ggmlv3.q2_K.bin',
                    model_type='llama', config=config)

# load the interpreted information from the local database
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
db = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)

# prepare a version of the llm pre-loaded with the local content
retriever = db.as_retriever(search_kwargs={'k': 2})
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

QA_LLM = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})


# In[14]:


def query(model, question):
    model_path = model.combine_documents_chain.llm_chain.llm.model
    model_name = pathlib.Path(model_path).name
    time_start = time.time()
    output = model({'query': question})
    response = output["result"]
    time_elapsed = time.time() - time_start
    display(HTML(f'<code>{model_name} response time: {time_elapsed:.02f} sec</code>'))
    display(HTML(f'<strong>Question:</strong> {question}'))
    display(HTML(f'<strong>Answer:</strong> {response}'))


# In[15]:


question = "What is the Q4 sales for BYD?"


# In[16]:


query(QA_LLM, question)


# In[19]:


question = "Who is the world's best selling EV?"
query(QA_LLM, question)


# In[20]:


question = "Who is the investor who backs BYD?"
query(QA_LLM, question)


# In[21]:


question = "How much EVs did BYD sell?"
query(QA_LLM, question)

