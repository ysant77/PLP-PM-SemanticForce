import pdfplumber
import pandas as pd

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})


def contains_pair(line):
  pairs = [("1", "business"), ("1a", "risk"), ("1b", "staff"), ("2", "properties"), ("3", "legal"),
           ("4", "safety"), ("5", "market"), ("6", "reserved"), ("7", "financial"), ("7a", "disclosure"),
           ("8", "financial"), ("9", "accountant"), ("9a", "control"), ("9b", "other"), ("9c", "disclosure"),
           ("10", "director"), ("11", "executive"), ("12", "security"), ("13", "certain"),
            ("14", "principal"), ("15", "exhibits"), ("16", "form")]

  pair_found = False
  for pair in pairs:
    if pair[0].lower() in line.lower() and pair[1].lower() in line.lower():
      pair_found = True
      break
  return pair_found

def extract_text_and_split_sections(pdf_path):
    # Initialize variables
    sections = {}
    current_section = None
    part_iv_found = False
    start_extraction = False

    # Open the PDF file and extract text line by line
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text().split('\n'):
                if line.lower().startswith('part iv'):
                  part_iv_found = True
                  continue
                if part_iv_found and line.lower().startswith('part i'):
                  start_extraction = True
                if start_extraction:
                  if line.startswith('Item') or line.startswith('ITEM'):
                    pair_found = contains_pair(line)
                    if pair_found:
                      current_section = line.split('.')[0].strip()
                      sections[current_section] = [line]
                    elif current_section:
                      sections[current_section].append(line)
                  elif current_section:
                    sections[current_section].append(line)
    data = [{'Item Number': item_number, 'Section Text': '\n'.join(section_content)}
            for item_number, section_content in sections.items()]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df

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



def generate_vector_db(text, output_path):
    
    db = FAISS.from_documents(text, embeddings)
    db.save_local(f"{output_path}/faiss")

def save(output_path, pdf_path):
   itemwise_df_10k = extract_text_and_split_sections(pdf_path)
   itemwise_df_10k.to_csv(f'{output_path}/data.csv', index=None)

   texts = split_document_into_text(pdf_path)
   generate_vector_db(texts, output_path)

