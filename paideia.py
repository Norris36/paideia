##
## Write a streamlit home page for my paideia project. The goal is to create a webstite that can help users learn about their subjects quicker
##
import streamlit as st
from datetime import datetime
import openai
# import tiktoken
import os
import time
import regex as re

def brute_force_sleep(wait = 5):
    now = int(datetime.now().timestamp())
    done = now + 5
    print("we're waiting")
    while now <= done:
        now = int(datetime.now().timestamp())
#brute_force_sleep()

# def num_tokens_from_string(string: str, encoding_name: str = 'cl100k_base') -> int:
#     """Returns the number of tokens in a text string."""
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# Write a function which takes a message in the format of a list, containing a dictionary with the keys "role" and "content", then ensure that the content is always stripped of any tabs, double spaces 

def message_cleaner(message):
    # message is a list containing a dictionary with the keys "role" and "content"
    # ensure that
    # 1. the content is always stripped of any tabs, double spaces
    # 2. the content is always stripped of any newlines
    # 3. the content is always stripped of any spaces at the beginning or end

    for i in range(len(message)):
        message[i]["content"] = message[i]["content"].replace("\n", "")
        message[i]["content"] = message[i]["content"].replace("\t", "")
        message[i]["content"] = message[i]["content"].replace("  ", "")
        message[i]["content"] = message[i]["content"].strip()

    return message

# write a function which takes a string as input, a long with a message, and adds the string as a dict with user and content as keys, the string as the value to content and then returns the key
def add_user_message(message, user_message):
    # message is a list containing a dictionary with the keys "role" and "content"
    # user_message is a string
    # add the user message to the message list, and return the message list
    message.append({"role":"user", "content":user_message})
    return message

# write a function which does the same as the above, but for the assistant
def add_assistant_message(message, assistant_message):
    # message is a list containing a dictionary with the keys "role" and "content"
    # assistant_message is a string
    # add the assistant message to the message list, and return the message list
    message.append({"role":"assistant", "content":assistant_message})
    return message

def set_summary_question(df, storage):
    new_index           = df[(df.last_run == df.last_run.min())& ~(df.summary.isna())].index.min()
    input_quesiton      = df.at[new_index, df.columns[5]]
    input_definition    = df.at[new_index, df.columns[4]]

    in_an_hour = int(datetime.now().timestamp() + 3600)

    st.session_state.question = ''


    filtered_df = storage[(storage[storage.columns[3]] == input_quesiton) & (storage[storage.columns[7]] > in_an_hour)]
    
    if len(filtered_df) > 2 or st.session_state.question == input_quesiton:
        new_index       = df[(df.last_run == df.last_run.min())& ~(df.summary.isna())].index.to_list()[1]
        input_quesiton      = df.at[new_index, df.columns[5]]
        input_definition    = df.at[new_index, df.columns[4]]

    st.session_state['new_index']       = new_index
    st.session_state['question']        = input_quesiton
    st.session_state['summary']         = input_definition
    #st.session_state['my_answer']       = ''


## show a dataframe on the site 
import pandas as pd
import numpy as np

openai.api_type = "azure"
openai.api_base = "https://jensbayopenaieastus.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
#openai.api_key = os.getenv('openai_key')
openai.api_key = "ff744e69396448808c16a4d5fadde7cc"


df = pd.read_csv('data.csv')
storage = pd.read_csv('storage.csv')

## Refactor the marked lines to a function setting the question and summary as session state variables
if 'new_line' not in st.session_state:
    st.session_state['new_line'] = set_summary_question(df, storage)
if 'my_reply' not in st.session_state:
    st.session_state['my_reply'] = ""
if 'my_answer' not in st.session_state:
    st.session_state['my_answer'] = ""

a_day_ago = datetime.now().timestamp() - 86400

progress = round((len(df[~(df.summary.isna()) & (df.last_run > a_day_ago)]) / len(df[~(df.summary.isna())]))* 100, 2) 

# Title
st.title(f"Paideia {progress} % ")



st.write(df[['last_run', 'last_score', 'question']].sort_values(by=['last_run'], ascending=False).head(10))


# Markdown
st.markdown(f"### This is a markdown {st.session_state['new_index']}")
st.write(st.session_state.question)

st.text_area("Answer", key="my_answer", value = st.session_state.my_answer)

qeustion_message = [{"role":"system",
                "content":"""
                I want you to act as a memory coach and exam teacher. 
                
                I will provide you with a question, my answer, and the answer of from the author. 
                My answer will be a summerisation of the authors answer, and your job is to evalute on a scale from 0/100 wether was correct or not, or how close.
                If my answer was in your opionion over 20/100 provide constructive feedback on how to improve the answer, what i missed.
                If my answer was below 20/100 or not at all close, say i answer i don't know or give me a hint, then provide me with a hint to the correct answer. 

                The format must always adhere to the following:
                your score / 100 
                
                Your feedback...
                """}]

with st.expander("Show me the storage"):
    st.selectbox("Select a question", storage[storage.columns[3]].unique(), key = 'storage_question')
    st.write(storage[storage[storage.columns[3]] == st.session_state['storage_question']])    

question = f"""
            The question im trying to answer is:
            {st.session_state.question}
            My answer is:   
            {st.session_state.my_answer}
            The authors answer is:
            {st.session_state.summary} """

if st.button('Get new question'):
    set_summary_question(df, storage)
add_user_message(qeustion_message, question)
st.session_state.new_line = st.session_state['new_index']
message = message_cleaner(qeustion_message)
if st.button('Submit'): 
    with st.spinner('Wait for it...'):
        response = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",
                    messages = message,
                    temperature=0.2,
                    max_tokens=350,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None)
        answer = response['choices'][0]['message']['content']
        st.session_state.my_reply = answer
    timestamp = int(datetime.now().timestamp())
    try:
        if len(re.findall("^0/100", answer))>0:
            score = 0
        elif len(re.findall("^100/100", answer))>0:
            score = 100
        else:
            score = int(re.findall("^(\d+?)/100", answer)[0])
            #score = int(re.findall("^(\d+?)/100", answer)[0])


    except:
        score = 1

    df.at[st.session_state.new_line, df.columns[8]] = score
    df.at[st.session_state.new_line, df.columns[9]] = timestamp
    #df.to_csv('data.csv', index=False)


    header = df.at[st.session_state.new_line, 'header']
    text = df.at[st.session_state.new_line, 'text']
    summary = st.session_state.summary
    question = st.session_state.question
    category = df.at[st.session_state.new_line, 'category']
    author = df.at[st.session_state.new_line, 'Author']
    new_line = [header, text, summary, question, category, author, score, timestamp]
    storage.loc[len(storage)] = new_line
    storage.to_csv('storage.csv', index=False)
    df.to_csv('data.csv', index=False)

    if score < 20:
        st.write('Too bad, try again!\n Read the hint and try again!')
    elif score < 50:
        st.write('Not bad, but you can do better!\n Read the hint and try again!')
    elif score < 70:
        set_summary_question(df, storage)
        st.write('Good job, but you can do better!\n Read the hint and try again!')
    
    st.write('My reply:', st.session_state.my_answer)

    st.markdown(f"> {st.session_state.my_reply}")

with st.expander('Show me the text'):
    st.write(st.session_state.summary)