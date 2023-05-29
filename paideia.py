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

def brute_force_sleep(wait = 5):
    now = int(datetime.now().timestamp())
    done = now + 5
    print("we're waiting")
    while now <= done:
        now = int(datetime.now().timestamp())
#brute_force_sleep()

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

def call_model(message):
    # This function takes a message as input, and returns the response from the model
    print('got here')
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
    print('got here to')
    st.session_state.my_reply = answer
    return answer

def set_summary_question():
    new_index = st.session_state.df.loc[st.session_state.df[~st.session_state.df.summary.isna()]['last_run'].idxmin()].name

    input_quesiton      = st.session_state.df.at[new_index, st.session_state.df.columns[5]]
    input_definition    = st.session_state.df.at[new_index, st.session_state.df.columns[4]]
    # except:
    #     input_quesiton      = st.session_state.df.at[new_index, st.session_state.df.columns[5]]
    #     input_definition    = st.session_state.df.at[new_index, st.session_state.df.columns[4]]
    in_an_hour = int(datetime.now().timestamp() + 3600)

    st.session_state.question = ''


    filtered_df = st.session_state.storage[(st.session_state.storage[st.session_state.storage.columns[3]] == input_quesiton) & (st.session_state.storage[st.session_state.storage.columns[7]] > in_an_hour)]
    
    if len(filtered_df) > 2 or st.session_state.question == input_quesiton:
        new_index       = st.session_state.df[(st.session_state.df.last_run == st.session_state.df.last_run.min())& ~(st.session_state.df.summary.isna())].index.to_list()[1]
        input_quesiton      = st.session_state.df.at[new_index, st.session_state.df.columns[5]]
        input_definition    = st.session_state.df.at[new_index, st.session_state.df.columns[4]]

    st.session_state['new_index']       = new_index
    st.session_state['question']        = input_quesiton
    st.session_state['summary']         = input_definition
    #st.session_state['my_answer']       = ''

def show_and_save_answer(answer):
    try:
        # Below we are trying to find the score, given by the model in the answer
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

        timestamp = int(datetime.now().timestamp())
        
        st.session_state.df.at[st.session_state.new_line, st.session_state.df.columns[8]] = score
        st.session_state.df.at[st.session_state.new_line, st.session_state.df.columns[9]] = timestamp
        #df.to_csv('data.csv', index=False)

        # Setting the values, for adding the date to the storage csv
        header      = st.session_state.df.at[st.session_state.new_line, 'header']
        text        = st.session_state.df.at[st.session_state.new_line, 'text']
        summary     = st.session_state.summary
        question    = st.session_state.question
        category    = st.session_state.df.at[st.session_state.new_line, 'category']
        author      = st.session_state.df.at[st.session_state.new_line, 'Author']
        new_line    = [header, text, summary, question, category, author, score, timestamp]
        st.session_state.storage.loc[len(st.session_state.storage)] = new_line
        st.session_state.storage.to_csv(get_storage_path(), index=False)
        st.session_state.df.to_csv(get_data_path(), index=False)

        if score < 20:
            st.write('Too bad, try again!\n Read the hint and try again!')
        elif score < 50:
            st.write('Not bad, but you can do better!\n Read the hint and try again!')
        elif score < 70:
            set_summary_question()
            set_summary_question()
            st.write('Good job, but you can do better!\n Read the hint and try again!')
        
        st.write('My reply:', st.session_state.my_answer)

        st.markdown(f"> {st.session_state.my_reply}")
    except Exception as e:
        print('show and save method')
        print(e)

def set_question():
    # The below code defines a function that returns a message and a question. The message provides
    # instructions for the user to provide a summary of an author's answer to a question, and the function
    # will evaluate the user's answer and provide feedback. The question variable contains placeholders
    # for the actual question, the user's answer, and the author's answer, which are retrieved from the
    # session state.
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

    question = f"""
            The question im trying to answer is:
            {st.session_state.question}
            My answer is:   
            {st.session_state.my_answer}
            The authors answer is:
            {st.session_state.summary} """
                
    return qeustion_message,question

def change_submitted():
    # The line of code `if st.session_state.submitted == True:` is checking if the value of the
    # `submitted` key in the `st.session_state` dictionary is equal to `True`. If it is, then the code
    # block inside the `if` statement will be executed. If it is not, then the code block will be
    # skipped.
    if st.session_state.submitted == True:
        st.session_state.submitted = False
    else:
        if len(st.session_state.my_answer) > 0:
            st.session_state.submitted = True
# write a function which gets the full path of the data.csv file
def get_data_path():
   # The `return os.path.join(os.path.dirname(__file__), "data.csv")` line of code is returning the
   # full path of the `data.csv` file by joining the directory path of the current file with the
   # filename `data.csv`.
   return os.path.join(os.path.dirname(__file__), "data.csv")

def get_storage_path():
   # The `return os.path.join(os.path.dirname(__file__), "storage.csv")` line of code is returning the
   # full path of the `storage.csv` file by joining the directory path of the current file with the
   # filename `storage.csv`.
   return os.path.join(os.path.dirname(__file__), "storage.csv")


openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          =  os.getenv("OPENAI_KEY")

st.session_state.df = pd.read_csv(get_data_path())
st.session_state.storage = pd.read_csv(get_storage_path())
## Refactor the marked lines to a function setting the question and summary as session state variables
if 'new_line' not in st.session_state:
    st.session_state['new_line'] = set_summary_question()
if 'my_reply' not in st.session_state:
    st.session_state['my_reply'] = ""
if 'my_answer' not in st.session_state:
    st.session_state['my_answer'] = ""
if 'df' not in st.session_state:
    df = pd.read_csv(get_data_path())
    st.session_state.df = df
if 'storage' not in st.session_state:
    storage = pd.read_csv(get_storage_path())
    st.session_state.storage = storage
if 'submitted' not in st.session_state:
    st.session_state.submitted = False


# Setting the progress value for the header
a_day_ago = datetime.now().timestamp() - 86400
progress = round((len(st.session_state.df[~(st.session_state.df.summary.isna()) & (st.session_state.df.last_run > a_day_ago)]) / len(st.session_state.df[~(st.session_state.df.summary.isna())]))* 100, 2) 

# Title
st.title(f"Paideia {progress} % ")
st.subheader("The number is the progress which you have made in the last 24 hours")

st.write(st.session_state.df[['last_run', 'last_score', 'question']].sort_values(by=['last_score', 'last_run']).head(10))

# Markdown
st.markdown(f"### This is a markdown {st.session_state['new_index']}")
st.write(st.session_state.question)


qeustion_message, question = set_question()

add_user_message(qeustion_message, question)
message = message_cleaner(qeustion_message)

st.text_area("Answer",
            key="my_answer",
            value = st.session_state.my_answer,
            on_change=change_submitted()
            )

with st.expander("Show me the storage"):
    st.selectbox("Select a question", st.session_state.storage[st.session_state.storage.columns[3]].unique(), key = 'storage_question')
    st.write(st.session_state.storage[st.session_state.storage[st.session_state.storage.columns[3]] == st.session_state['storage_question']])    
    #st.write(st.session_state.storage.sort_values('last_run', ascending=False))
if st.button('Get new question'):
    set_summary_question()

st.session_state.new_line = st.session_state['new_index']

if st.button('Submit') or st.session_state.submitted: 
    with st.spinner('Wait for it...'):
        answer = call_model(message)
    show_and_save_answer(answer)    

with st.expander('Show me the text'):
    st.write(st.session_state.summary)