from dotenv import load_dotenv
load_dotenv()
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

st.header("My Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")


# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if len(chunks) == 0:
        st.error("PDF produced zero text.")
        st.stop()

    # Embeddings (free)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    # Vector DB
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User input
    user_question = st.text_input("Type your question here")

    if user_question:
        match = vector_store.similarity_search(user_question)

        # Groq LLM
        LLM = ChatGroq(
            groq_api_key=os.environ["GROQ_API_KEY"],
            model_name="llama-3.3-70b-versatile",   
            temperature=0,
            max_tokens=1000
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the following context:\n{context}\nQuestion: {question}"
        )

        # Chain
        chain = (
            {
                "context": lambda x: "\n\n".join(
                    doc.page_content for doc in x["input_documents"]
                ),
                "question": RunnablePassthrough()
            }
            | prompt
            | LLM
        )

        # Response
        response = chain.invoke({
            "input_documents": match,
            "question": user_question
        })

        st.write(response.content)
