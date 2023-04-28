import os
from apiKey import apiKey

# Streamlit is application framework to allow us to work with different services
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

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
    input_variables = ['title', 'wikipedia_research'],
    template="Write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research} "
)

# MEMORY
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# CREATE INSTANCE OF OUR LLM SERVICE
llm = OpenAI(temperature=0.9)
# chain multiple llms together by langchain
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wikipedia = WikipediaAPIWrapper()

# we need to join chains together so they can communicate

# print stuff to the screen if there is a prompt
if prompt:
    title = title_chain.run(prompt)
    wikipedia_research = wikipedia.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wikipedia_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.write(title_memory.buffer)

    with st.expander('Script history'):
        st.write(script_memory.buffer)

    with st.expander('Wikipedia research'):
        st.write(wikipedia_research)

