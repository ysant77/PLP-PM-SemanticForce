from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

import os
import time
import pathlib
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".
    Context: {context}
    Question: {question}
    Answer:
    """

config = {'max_new_tokens': 1024, 'temperature': 0.0}
llm = CTransformers(model='./models/llama-2-7b-chat.ggmlv3.q2_K.bin',
                    model_type='llama', config=config)

QA_LLM = None

def split_document_into_text(pdf_path):
    pdf_path_arr = pdf_path.split("/")
    pdf_name = pdf_path_arr[-1].strip()
    data_dir = "/".join(pdf_path_arr[:-1])
    loader = DirectoryLoader(path=data_dir, glob=pdf_name, loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=50)
    texts = splitter.split_documents(documents)
    return texts



def generate_vector_db(text):
    
    db = FAISS.from_documents(text, embeddings)
    db.save_local("faiss")

def generate_qa_llm_model():
    global QA_LLM
    db = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={'k': 2})
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])

    QA_LLM = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})    

def query(model, question):
    model_path = model.combine_documents_chain.llm_chain.llm.model
    time_start = time.time()
    output = model({'query': question})
    response = output["result"]
    return response