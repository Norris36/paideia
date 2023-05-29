##
## Help ne writing a streamlit app to search and have the full overview over the data.csv
## Is this udnerstood?
##

import streamlit as st
import pandas as pd
import os 
def get_data_path():
   return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data.csv")

def check_text_length(text):
    return 200 if not isinstance(text, str) else int(len(text)/4)


if 'df' not in st.session_state:
   df = pd.read_csv(get_data_path())
   #df.dropna(inplace=True)
   st.session_state.df = df
if 'cols' not in st.session_state:
   col = st.session_state.df.columns
   col = ['last_run', 'last_score', 'question']
   st.session_state['cols'] = col
if 'index' not in st.session_state:
   if len(st.session_state.df[st.session_state.df.question.isna()].index.tolist()) > 0:
      index = st.session_state.df[st.session_state.df.question.isna()].index.tolist()[0]
   else:
      index = 0
   st.session_state.index = index

st.multiselect('chose which columns to see in the datafra',
               options=st.session_state.df.columns,
               default=st.session_state.cols,
               key = st.session_state.cols)


st.dataframe(st.session_state.df[st.session_state.cols].sort_values('last_run'),
             width=6000,
             height=1000)

st.write('Explore specific items')

selected_index = st.selectbox(
    'choose a specific row to update it',
    options = st.session_state.df.index,
    key     = 'selected_index',
    index   = int(st.session_state.index)
)

# Update the session state index when the select box changes
if st.session_state.index != selected_index:
    st.session_state.index = selected_index

st.text_input(
    'Want to modify the question?',
    value = st.session_state.df.at[st.session_state.index, 'question'],
    key = 'question'
)

st.text_area(
    'Want to modify the summary?',
    value = st.session_state.df.at[st.session_state.index, 'summary'],
    height = check_text_length(st.session_state.df.at[st.session_state.index, 'summary']),
    key = 'summary'
)

st.text_area(
    'Want to modify the text?',
    value   = st.session_state.df.at[st.session_state.index, 'text'],
    height  = check_text_length(st.session_state.df.at[st.session_state.index, 'text']),
    key = 'text'
)

if st.button('update values in dataframe'):
    st.session_state.df.at[st.session_state.index, 'question'] = st.session_state.question
    st.session_state.df.at[st.session_state.index, 'text'] = st.session_state.text
    st.session_state.df.at[st.session_state.index, 'summary'] = st.session_state.summary
    st.session_state.df.to_csv(get_data_path(), index=False)

if st.button('Delete selected row'):
    st.session_state.df = st.session_state.df.drop(st.session_state.index)
    st.session_state.df.to_csv(get_data_path(), index=False)
