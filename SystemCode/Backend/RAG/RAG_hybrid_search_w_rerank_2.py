# Third-party Libraries
from langchain import hub
from langchain_core.documents import Document
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
import google.generativeai as genai
import wikipedia
import mistune
import dotenv

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# Native Libraries
import requests
import html
import os
import re
import pickle as pkl

# LLM Caching
set_llm_cache(InMemoryCache())

# Load Variables
dotenv.load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
home_dir = os.getcwd() # this directory is where you start running python script from
curr_dir = os.path.join(home_dir, 'Backend/RAG')


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# ======== WIKIPEDIA FETCHING DOCUMENTS ========
def fetch_wikipedia_page(title):
    try:
        page = wikipedia.page(title)
    except:
        page = wikipedia.page(title, auto_suggest=False)
    return page.content

# wikipedia.exceptions.PageError
def remove_markdown(text):
    return mistune.html(text)  # Converts markdown to plain text

def remove_references(text):
    return re.sub(r"\[[\d\s]+\]", "", text)  # Basic reference removal

def prepare_wikipedia_document(wikipedia_title):
    raw_text = fetch_wikipedia_page(wikipedia_title)
    if raw_text:
        cleaned_text = remove_markdown(raw_text)
        cleaned_text = remove_references(cleaned_text) 
        return {"text": cleaned_text}
    else:
        return None


def RAG_setup():

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
    #                                             chunk_overlap=50,
    #                                             length_function=len,
    #                                             is_separator_regex=False
    #                                             )

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # ======== PREPARATION OF VECTOR EMBEDDINGS ========
    # company_name_file = open('10K_report_company_names.txt','r')
    # Lines = company_name_file.readlines()
    # chunks_3 = []
    # for x,line in enumerate(Lines):
    #     # print(x, line)
    #     chunks_3.extend([Document(page_content=html.unescape(i)) for i in text_splitter.split_text(prepare_wikipedia_document(line.replace('\n',''))['text'])])

    # Save vector embeddings
    # vectorstore_3 = Chroma.from_documents(
    #                      documents=chunks_3,                 # Data
    #                      embedding=gemini_embeddings,    # Embedding model
    #                      persist_directory="./chroma_db_3" # Directory to save data
    #                      )

    # # load vector embeddings from disk
    # vectorstore_disk_3 = Chroma(
    #                         persist_directory="./chroma_db_3",       # Directory of db
    #                         embedding_function=gemini_embeddings   # Embedding model
    #                 )


    # ======== RETRIEVER: initialize the bm25 retriever and faiss retriever (HYBRID SEARCH) ========
    # chunk_length = len(chunks_3)
    # docu_chunk_arr = [chunk.page_content for chunk in chunks_3]

    # # sparse vector database
    # bm25_retriever = BM25Retriever.from_texts(docu_chunk_arr) # , metadatas=[{"source": 1}] * chunk_length
    # bm25_retriever.k = 5
    # # to save
    # with open('bm25result', 'wb') as bm25result_file:
    #     pkl.dump(bm25_retriever, bm25result_file)

    #to read bm25 object
    bm25_retriever = BM25Retriever
    with open(f'{curr_dir}/bm25result', 'rb') as bm25result_file:
        bm25_retriever = pkl.load(bm25result_file)

    # # dense vector database
    # faiss_vectorstore = FAISS.from_texts(docu_chunk_arr, gemini_embeddings) # , metadatas=[{"source": 2}] * chunk_length
    faiss_vectorstore = FAISS.load_local(f"{curr_dir}/faiss_index", gemini_embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])


    # # ======== AUGMENT ========
    # template = """You are an assistant for question-answering tasks. 
    # Use the following pieces of retrieved context to answer the question. 
    # If you don't know the answer, just say that you don't know. 
    # Use three sentences maximum and keep the answer concise.
    # Question: {question} 
    # Context: {context} 
    # Answer:
    # """
    # prompt = ChatPromptTemplate.from_template(template)


    # ======== GENERATE ========
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.70, top_k=15)
    compressor = FlashrankRerank()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    # chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

    return compression_retriever

def RAG_query(compression_retriever, query):
    # query = 'Provide me a short timeline of the history of Goldman Sachs.'
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.70, top_k=15)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
    res = chain.invoke(query)

    return res['result']
