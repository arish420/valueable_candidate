import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore  # ✅ Correct
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import tiktoken

# Load environment variables
load_dotenv()
###############################################################setting openai ai api##################################################################

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

def get_prompt(converasation):
    return f"""You are an AI assistant designed to assess whether a candidate is suitable for manual labor professions based on their Facebook Messenger conversation. Follow these steps strictly:
    
    NOTE:
    - The conversation is in **Polish language**, so translate and analyze carefully.
    - If the **sender name is 'Ryszard Konieczny'**, IGNORE his messages — he is the recruiter (me).
    - ONLY analyze the messages from the responder (candidate).
    
    I. ASSESS SUITABILITY (Verify Criteria Met):
    From the conversation, determine whether the candidate satisfies the following aspects (not all may be present):
    1. Possession of Construction Tools
    2. Own Accommodation
    3. Own Transport
    4. Communicative English Language Skills
    5. Work Experience:
       - Company names (if discussed)
       - Years of experience
       - Relevance to the profession they are applying for
    
    Important Rules:
    - If only one aspect is discussed and it is met, add to database.
    - If two aspects are discussed and one is not met, reject the candidate.
    - If three or more aspects are discussed, evaluate overall suitability.
    - If the candidate clearly mentions a logistical obstacle (e.g., lives too far), still add to database.
    
    II. PROFESSION CLASSIFICATION:
    If criteria from Point I are met, classify the candidate under one of the following professions:
    - Electrician
    - Plumber
    - Carpenter
    - Concrete Specialist
    - Painter
    - CNC Operator
    - Concrete Repairer
    - Steel Structure Installer
    - Roofer
    - MetalStud Installer
    - Ceiling System Installer
    - Earthworker
    - Plasterer
    - Window Installer
    - Tiler
    
    III. DATA EXTRACTION:
    If the candidate passes Point I:
    - Extract Full Name (if mentioned)
    - Extract Phone Number (if mentioned)
    - Extract Place of Residence (if mentioned)
    - Extract Profession (as per classification)
    
    IV. INPUT CONVERSATION FOR ANALYSIS:
    Now analyze the following conversation based on the above rules.
    The conversation is in Polish and between the recruiter (Ryszard Konieczny) and the candidate in form json.
    IGNORE messages from Ryszard Konieczny. Analyze only the responder.
    
    Conversation:
    {converasation}
    """




st.title("Get Valuable Candidate")

file=st.file_uploader("Import File",type=['json'])
# st.write()
# raw_text=""
# with open(file,'r') as f:
#     for line in f:
#         raw_text = " " + line
raw_text=uploaded_file.getvalue().decode('utf-8')
if st.button("Yes"):
    st.write(raw_text)


