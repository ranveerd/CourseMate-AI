import streamlit as st
import tempfile
import os
import shutil
import hashlib

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("📚 Book RAG Assistant")
st.write("Upload a PDF book and ask questions based on its content.")

# -----------------------------------
# Embedding model
# -----------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -----------------------------------
# Upload PDF
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF book",
    type=["pdf"]
)

if uploaded_file is not None:

    # Create a unique ID for the uploaded PDF
    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    db_path = os.path.join(
        "chroma-db",
        file_hash
    )

    # -----------------------------------
    # Create database only once
    # -----------------------------------

    if not os.path.exists(db_path):

        with st.spinner("Processing your book..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            # Create Chroma database
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=db_path
            )

            # Remove temporary PDF
            os.remove(pdf_path)

        st.success("✅ Book uploaded and processed successfully!")

    else:

        # Load existing database
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )

        st.success("✅ Book is ready!")

    # -----------------------------------
    # Retriever
    # -----------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # -----------------------------------
    # Gemini LLM
    # -----------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # -----------------------------------
    # Prompt
    # -----------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use only the provided context to answer the question.

If the answer is not present in the context, say:
"I could not find the answer in the provided book."

Give a clear, natural and properly formatted answer.
Do not mention the context, retrieved documents, or RAG system."""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}"""
            )
        ]
    )

    # -----------------------------------
    # Ask Question
    # -----------------------------------

    query = st.text_input(
        "Ask a question about the book:"
    )

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )

        response = llm.invoke(final_prompt)

        # Extract clean Gemini response
        if isinstance(response.content, list):

            answer = ""

            for item in response.content:

                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                ):
                    answer += item.get("text", "")

        else:
            answer = response.content

        st.write("### 🤖 AI Answer")
        st.write(answer)