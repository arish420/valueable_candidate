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
import json
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


def get_conversation(conversation):
    return f""" You are provided with a Facebook Messenger conversation in raw text JSON format.

    Each message includes a 'sender' (the person who sent the message) and its 'content' (the actual message text). The conversation is in **Polish**.
    
    Your task is to:
    
    1. Translate each 'content' value from **Polish to English**.
    2. For each message, return a **single-line output** in the format:
       Sender: Translated Message
    3. Keep the order of messages as in the original JSON.
    4. Do not change or omit any message.
    5. If a message is empty or non-textual (e.g., photo, sticker), skip it.
    6. No additional explation from your side.
    
    Here is the JSON conversation:
    
    {conversation}
    
    Return each message on a **separate line**, like:
    Sender Name: Translated Message
    Sender Name: Translated Message
    ...
    
    """
    
def get_prompt(converasation):
    return f"""You are an AI assistant designed to assess whether a candidate is suitable for manual labor professions based on their Facebook Messenger conversation. Follow these steps strictly:
    
    NOTE:
    
    - If the **sender name is 'Ryszard Konieczny'**, IGNORE his messages — he is the recruiter (me).
    - ONLY analyze the messages from the responder (candidate).
    
    I. ASSESS SUITABILITY (Verify Criteria Met):
    From the conversation, determine whether the candidate satisfies the following aspects (not all may be present):
    1. Possession of Construction Tools
    2. Own Accommodation
    3. Own Transport
    4. Communicative English Language Skills (response should be binary, either person have English communication or not) no your suggestion
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

    # Valuebale: Yes [if candiate possess all the criteria], otherwise No


    
    Conversation:
    {converasation}
    """




st.title("Get Valuable Candidate")

uploaded_file=st.file_uploader("Import File",type=['json'])
# st.write()
# raw_text=""
# with open(file,'r') as f:
#     for line in f:
#         raw_text = " " + line
if st.button("Find Candidate"):
    raw_text=uploaded_file.getvalue().decode('utf-8')

    res=llm_llama3.invoke(get_conversation(raw_text))
    res=llm_llama3.invoke(get_prompt(res.content))
    st.write(res.content)
    # st.write(json.dump(raw_text))


