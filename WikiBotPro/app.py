from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pyttsx3

import wikipediaapi
import urllib.parse
import warnings

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

warnings.filterwarnings("ignore")
wiki_api = wikipediaapi.Wikipedia("Custom Wiki AI Chatbot", "en")


def main():
    load_dotenv()
    st.set_page_config(page_title="Wiki Chatbot")
    st.header("Wiki ChatBot 🌎")
    st.markdown(hide_default_format, unsafe_allow_html=True)

    def get_keyword_from_url(url):
        try:
            parsed_url = urllib.parse.urlparse(url)
            path_segments = parsed_url.path.split("/")
            # Remove any empty segments
            path_segments = [segment for segment in path_segments if segment]

            if path_segments:
                keyword = path_segments[-1]
                return keyword
            else:
                return None
        except Exception as e:
            st.error("Invalid URL. Please enter a valid Wikipedia URL.")
            return None

    # upload file
    user_url = st.text_input("Enter Wikipedia URL: ")
    keyword = get_keyword_from_url(user_url)

    if "current_keyword" not in st.session_state or st.session_state.current_keyword != keyword:
        st.session_state.current_keyword = keyword
        # Clear the chat window when a new Wikipedia URL is entered
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    wiki_tag = wiki_api.page(keyword)
    if wiki_tag.exists() == True:
        print("Keyword:", keyword)
    else:
        st.error("Invalid Wikipedia URL. Please enter a valid Wikipedia URL.")
        st.stop()  # Stop the execution if the URL is invalid.

    if keyword is not None:
        wiki_text = wikipediaapi.Wikipedia(
            user_agent="Custom Wiki AI Chatbot",
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
        wiki_full_text = wiki_api.page(keyword)
        text_data = wiki_full_text.text

        def save_text_as_pdf(text_data, output_file):
            doc = SimpleDocTemplate(output_file, pagesize=letter)
            styles = getSampleStyleSheet()
            content = [Paragraph(text_data, styles["Normal"])]
            doc.build(content)

        output_file = (
            "Training_Data.pdf"  # Replace with the desired output PDF file name
        )
        save_text_as_pdf(text_data, output_file)

        # extract the text
        with open("Training_Data.pdf", "rb") as file:

            pdf_reader = PdfReader(file)
            num_pages = len(pdf_reader.pages)
            text = ""

            for page_number in range(num_pages):
                page = pdf_reader.pages[page_number]
                text += page.extract_text()

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            # show user input
            engine = pyttsx3.init()
            if prompt := st.chat_input("Ask your question: "):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                docs = knowledge_base.similarity_search(prompt)
                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=prompt)

                with st.chat_message("assistant"):
                    st.markdown(response)
                    engine.say(response)
                try:
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except AttributeError:
                    st.markdown(response)
                try:
                    engine.runAndWait()
                except RuntimeError:
                    pass


if __name__ == "__main__":
    main()
