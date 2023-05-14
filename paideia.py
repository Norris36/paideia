##
## Write a streamlit home page for my paideia project. The goal is to create a webstite that can help users learn about their subjects quicker
##

import streamlit as st

# Title
st.title("Paideia")

# Header
st.header("Welcome to Paideia")

# Subheader
st.subheader("A website to help you learn about your subjects quicker")

# Text
st.text("This is a text")

# Markdown
st.markdown("### This is a markdown")

## show a dataframe on the site 
import pandas as pd
import numpy as np

# Create a function which reads the dataframe if it doesn't exist and then returns it, if it exists, doesn't do anything
@st.cache_data
def load_data():

    df = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['a', 'b', 'c']
    )
    return df

df = load_data()

st.dataframe(df,
             width= 1000,
             height=410)  # Same as st.write(df)

## write a select function which can modify numbers in the dataframe, in column 2

# selectbox
# mod this so that you get a list of the index, and then get an input slider between 0-100
option = st.selectbox(
    'Which number do you want to modify?',
        df.index)


st.write('You selected: ', option    )

# slider
value_option = st.slider( 
    'Which number do you want to modify?',
        0, 100, 50)

# modify the dataframe with the new value, on a button click
if st.button('Modify'):
    df.iloc[option,     2] = value_option

st.dataframe(df)