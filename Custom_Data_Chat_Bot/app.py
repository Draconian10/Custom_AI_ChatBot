import streamlit as st
import langchain
#from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import openai
from langchain.llms import HuggingFaceHub
import os
from PyPDF2 import PdfReader
from docx import Document
import toml

def main():
    #load_dotenv()
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title="Chat with multiple files", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple files :books:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("What is up?")
    if user_question:
        if st.session_state.conversation is None:
            st.error("Please upload files and click 'Process' before asking questions.")
        else:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    selected_tab = st.sidebar.radio("Navigation", options=["Files", "Text"], horizontal=True, label_visibility="collapsed")

    if selected_tab == "Files":
        st.sidebar.subheader("Upload and Process Files")
        uploaded_files = st.sidebar.file_uploader("Upload your files here and click on 'Process'", accept_multiple_files=True)

        if uploaded_files:
            total_character_count = sum(len(file.read()) for file in uploaded_files)
            if total_character_count > 400000:
                st.warning("Total input data should not exceed 400,000 characters.")
                st.stop()

            if st.sidebar.button("Process"):
                with st.spinner("Processing"):
                    st.session_state.conversation = None
                    st.session_state.chat_history = None

                    text = ""
                    for file in uploaded_files:
                        file_extension = os.path.splitext(file.name)[1].lower()

                        if file_extension == '.pdf':
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                        elif file_extension == '.txt':
                            text += file.read().decode("utf-8")
                        elif file_extension == '.docx':
                            doc = Document(file)
                            for paragraph in doc.paragraphs:
                                text += paragraph.text + "\n"
                        else:
                            st.warning('We only support PDF, TXT and DOCX files')
                            st.stop()

                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                    text_chunks = text_splitter.split_text(text)

                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

                    llm = ChatOpenAI()
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
                    st.session_state.conversation = conversation_chain

    elif selected_tab == "Text":
        st.sidebar.subheader("Enter Text")
        user_text = st.sidebar.text_area("Enter your text here", "")

        if st.sidebar.button("Process Text"):
            # Process the user's entered text
            if user_text:
                total_character_count = len(user_text)
                if total_character_count > 400000:
                    st.warning("Total input data should not exceed 400,000 characters.")
                    st.stop()

                st.session_state.conversation = None
                st.session_state.chat_history = None

                text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                text_chunks = text_splitter.split_text(user_text)

                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

                llm = ChatOpenAI()
                memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
                st.session_state.conversation = conversation_chain

if __name__ == '__main__':
    main()
