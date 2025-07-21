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
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.7)


# GROQ_API_KEY=os.getenv("GROQ_API_KEY")


os.environ["GROQ_API_KEY"] =  df_groq.keys()[0]


from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    # api_key=GROQ_API_KEY
)


def get_conversation(conversation):
    return f"""
        You are an AI assistant evaluating whether a candidate is suitable for a manual labor job based on a Facebook Messenger conversation.
        
        The conversation is provided in JSON format where each message includes:
        - 'sender': name of the person who sent the message
        - 'content': the message text
        The conversation is in **Polish**.
        
        ==========================
        I. MESSAGE FILTERING RULES:
        ==========================
        1. IGNORE messages from **Ryszard Konieczny** (the recruiter).
        2. Only analyze messages from the candidate.
        3. Carefully interpret and translate Polish messages as needed.
        
        ==========================
        II. ASSESS SUITABILITY:
        ==========================
        Evaluate whether the candidate meets the following criteria:
        
        1. **Possession of Construction Tools**:
           - List any tools mentioned by the candidate.
        
        2. **Own Accommodation**:
           - Extract the location of accommodation if mentioned.
        
        3. **Own Transport**:
           - Mention the type of transport (e.g. car, van, public transport).
        
        4. **English Communication Skills**:
           - Does the candidate say they can speak/understand English?
           - Answer: Yes or No
        
        5. **Work Experience**:
           - Mention any companies referenced.
           - Note duration (years/months) of experience.
           - Identify whether experience is relevant to the job role.
        
        ==========================
        III. SUITABILITY DECISION:
        ==========================
        Based on the above criteria:
        
        - If only one aspect is discussed and it is met → **Add to database**
        - If two aspects are discussed and one is NOT met → **Reject the candidate**
        - If three or more aspects are discussed → **Evaluate overall suitability**
        - If a logistical challenge is mentioned (e.g. far away residence) → **Still add to database**
        
        ==========================
        IV. PROFESSION CLASSIFICATION:
        ==========================
        If the candidate is suitable, classify them into one of the following roles:
        
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
        
        Also return:
        - **Valuable**: Yes → if candidate satisfies all criteria
        - **Valuable**: No → if any key criteria are missing
        
        ==========================
        V. CANDIDATE DETAILS (If valuebale == Yes):
        ==========================
        Extract the following if mentioned:
        - Full Name
        - Phone Number
        - Place of Residence
        - Profession (as classified above)
        
        ==========================
        VI. INPUT CONVERSATION:
        ==========================
        Below is the conversation in JSON format (in Polish). Ignore recruiter messages. Analyze only candidate's messages.
        
        Conversation:
        {conversation}
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
    res=llm.invoke(get_prompt(raw_text))
    st.write(res.content)
    # st.write(json.dump(raw_text))


