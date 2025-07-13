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



# https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df_openai=pd.read_csv(url)
# st.write(df)


# def download_db():
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
# k=""
# with open(download_db(),'r') as f:
#     f=f.read()
#     # st.write(f)
#     k=f
# # st.write(k)
os.environ["OPENAI_API_KEY"] =  df_openai.keys()[0]


sheet_id = '1iFqeoyyDg2zkkJiTv_GXm0xRcUXn5SQ05c-XfHull_Q' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df_groq=pd.read_csv(url)

# https://docs.google.com/spreadsheets/d/1iFqeoyyDg2zkkJiTv_GXm0xRcUXn5SQ05c-XfHull_Q/edit?gid=0#gid=0



# Initialize GPT-4o-mini model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)


# GROQ_API_KEY=os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] =  df_groq.keys()[0]


from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    # api_key=GROQ_API_KEY
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
    - The conversation is in Polish and between the recruiter (Ryszard Konieczny) and the candidate in form json.
    - Decode the letter codes with great attention in context to polish language.
    
    
    I. ASSESS SUITABILITY (Verify Criteria Met):
    From the conversation, determine whether the candidate satisfies the following aspects (not all may be present):
    1. Possession of Construction Tools : Return mentioned tools
    2. Own Accommodation: Mention location of accommodation
    3. Own Transport: Mention the transport namme
    4. Communicative English Language Skills: Does candidate mention english communication skill in chat? Yes or No based on the conversation
    5. Work Experience:
       - Company names (if discussed)
       - Years of experience: Yes and mention year/months
       - Relevance to the profession they are applying for
    
    Important Rules:
    - If only one aspect is discussed and it is met, add to database.
    - If two aspects are discussed and one is not met, reject the candidate.
    - If three or more aspects are discussed, evaluate overall suitability.
    - If the candidate clearly mentions a logistical obstacle (e.g., lives too far), still add to database.
    
    II. PROFESSION CLASSIFICATION: Return relevant profession
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

    

    # Valuebale: Yes [if candiate possess all the criteria], otherwise No

    III. DATA EXTRACTION:
    If the candidate passes "ASSESS SUITABILITY" in Point I:
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

uploaded_file=st.file_uploader("Import File",type=['json'])
# st.write()
# raw_text=""
# with open(file,'r') as f:
#     for line in f:
#         raw_text = " " + line
if st.button("Find Candidate"):
    raw_text=uploaded_file.getvalue().decode('utf-8')
    # st.write(raw_text)

    # # res=llm.invoke(get_conversation(raw_text))
    res=llm_llama3.invoke(get_prompt(raw_text))
    st.write(res.content)
    # st.write(json.dump(raw_text))


