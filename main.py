import os
import streamlit as st
import pickle
import time
import langchain
from langchain.llms import GooglePalm
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

st.title("News Research Tool")
st.sidebar.title("New Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()
llm = GooglePalm(temperature=0.9, max_tokens=500)

if process_url_clicked:
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    embeddings = HuggingFaceHubEmbeddings()
    vectorstore_db = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_db, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain((query), return_only_outputs=True)
            st.header("Answer")
            st.write(result)

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)



    



