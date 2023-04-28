import os
from apiKey import apiKey

# Streamlit is application framework to allow us to work with different services
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

os.environ['OPENAI_API_KEY'] = apiKey

# APP FRAMEWORK
st.title('Youtube GPT Creator')
# place where we gonna prompt to llm
prompt = st.text_input('Plug in your prompt here')


# PROMPT TEMPLATES
title_template = PromptTemplate(
    input_variables = ['topic'],
    template="Write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template="Write me a youtube video script based on this title TITLE: {title}"
)

# CREATE INSTANCE OF OUR LLM SERVICE
llm = OpenAI(temperature=0.9)
# chain multiple llms together by langchain
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script')
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],  output_variables=['title', 'script'], verbose=True)

# we need to join chains together so they can communicate

# print stuff to the screen if there is a prompt
if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title'])
    st.write(response['script'])

