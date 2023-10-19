import pickle

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os
import time

file_path = 'vector-openai.pkl'
# Set your OpenAI API key
openai_api_key = "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"

# Set your HuggingFace Hub API token
huggingface_api_token = "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"


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
#
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     # conversation_chain = ConversationalRetrievalChain.from_llm(
#     #     llm=llm,
#     #     retriever=vectorstore.as_retriever(),
#     #     memory=memory
#     # )
#     return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title="Constitutions Chatbot",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    progress_placeholder=st.empty()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Constitutions Chatbot :books:")
    user_question = st.text_input("Ask a question about your pdfs:")
    # if user_question:
    #     time.sleep(5)
    #     handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents/Pdfs")
        pdf_docs = st.file_uploader(
            "Upload Pdfs and let's start with ConstitutionsChatbot", accept_multiple_files=True)
        if st.button("Proceed"):
            with st.spinner("Loading..."):
                time.sleep(5)
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                progress_placeholder.text("Text chunks.....")
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                progress_placeholder.text("Create vectorstore.....")
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                progress_placeholder.text("save vectorstore to pkl file.....")
                with open(file_path, 'wb') as f:
                    pickle.dump(vectorstore)

                # time.sleep(5)
                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)
                if user_question:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            vectorsStore = pickle.load(f)
                            llm = ChatOpenAI()
                            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                            chain = RetrievalQAWithSourcesChain.from_llm(
                                llm=llm,
                                retriever=vectorsStore.as_retriever(),
                                memory=memory
                            )
                            results = chain({'Question': user_question}, return_only_outputs=True)
                            st.header("Answers: ")
                            st.write(results['answers'])


if __name__ == '__main__':
    main()
