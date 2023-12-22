import streamlit as st
import langchain
# from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import openai
import os
from PyPDF2 import PdfReader
from docx import Document
import pinecone
import time
from langchain.vectorstores import Pinecone
import toml

def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple files", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone_env = st.secrets["PINECONE_ENV"]
    index_name = st.secrets["PINECONE_INDEX_NAME"]

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
            st.error("Please provide data and click 'Process' before asking questions.")
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
            if st.sidebar.button("Process"):
                with st.spinner("Processing"):

                    # initialize pinecone
                    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

                    if index_name in pinecone.list_indexes():
                        pinecone.delete_index(index_name)

                    # we create a new index
                    pinecone.create_index(name=index_name, metric='cosine',
                                          dimension=1536)  # 1536 dim of text-embedding-ada-002

                    # wait for index to be initialized
                    while not pinecone.describe_index(index_name).status['ready']:
                        time.sleep(1)

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
                    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)
                    llm = ChatOpenAI(model_name = 'gpt-4')
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
                    st.session_state.conversation = conversation_chain

    elif selected_tab == "Text":
        st.sidebar.subheader("Enter Text")
        user_text = st.sidebar.text_area("Enter your text here", "")

        if st.sidebar.button("Process Text"):
            if not user_text.strip():
                st.warning("Please enter some text before processing.")
            else:
                # Process the user's entered text
                if user_text:
                    # total_character_count = len(user_text)
                    # if total_character_count > 400000:
                    #     st.warning("Total input data should not exceed 400,000 characters.")
                    #     st.stop()

                    st.session_state.conversation = None
                    st.session_state.chat_history = None

                    # initialize pinecone
                    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

                    if index_name in pinecone.list_indexes():
                        pinecone.delete_index(index_name)

                    # we create a new index
                    pinecone.create_index(name=index_name, metric='cosine',
                                          dimension=1536)  # 1536 dim of text-embedding-ada-002

                    # wait for index to be initialized
                    while not pinecone.describe_index(index_name).status['ready']:
                        time.sleep(1)

                    st.session_state.conversation = None
                    st.session_state.chat_history = None

                    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
                    text_chunks = text_splitter.split_text(user_text)

                    embeddings = OpenAIEmbeddings()
                    vectorstore = Pinecone.from_texts(text_chunks, embeddings, index_name=index_name)

                    llm = ChatOpenAI(model_name = 'gpt-4')
                    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
                    st.session_state.conversation = conversation_chain

if __name__ == '__main__':
    main()
