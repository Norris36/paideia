##
## Write a streamlit home page for my paideia project. The goal is to create a webstite that can help users learn about their subjects quicker

import streamlit as st
from datetime import datetime
import openai
import os
import time
import regex as re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          =  os.getenv("OPENAI_KEY")

### Here we set all the appropiate values

def get_french_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(os.path.dirname(__file__)), "frenchmen.csv")

### Here we are going to set all the session state stuff

if 'objective' not in st.session_state:
    st.session_state.objective = "Practice Translating simple sentences, French to English"
if 'df' not in st.session_state:
    df = pd.read_csv(get_french_path())
    st.session_state.df = df
if 'sample' not in st.session_state:
    sample = df.sort_values('last_run').head(10)
    st.session_state.sample = sample
if 'index' not in st.session_state:
    # can you ensure the index is set to a random index value in the sample dataframe?
    index = st.session_state.sample.index.tolist()[0]
    st.session_state.index = index
if 'query' not in st.session_state:
    query = st.session_state.sample.at[st.session_state.index, 'French Sentence']
    st.session_state.query = query
if 'model' not in st.session_state:
    model = ""
    st.session_state.model = model
if 'submitted' not in st.session_state:
    submitted = False
    st.session_state.submitted = submitted


# please tell me when you wake om my frien ; )

st.title("French Teacher")

tasks = [
    "Practice Translating simple sentences, French to English",
    "Practice Translating simple sentences, English to French",
]

st.selectbox('What is the purpose of your visit?',
             key = 'objective',
             options=tasks)

st.dataframe(st.session_state.sample[['French Sentence', 'English Translation', 'last_run']].sort_values('last_run'))

st.markdown('## French to English')
st.markdown('### Original Lanugage')
#st.write(st.session_state.query)
st.text_area('Translation',
                key = 'translation',)
