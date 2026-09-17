from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory= "chroma-db",
    embedding_function= embedding_model
)

retriever = vector_store.as_retriever(
    search_type= "mmr",
    search_kargs= {
        "k" : 4,
        "fetch-k" : 10,
        "lambda_mult" : 0.5
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful AI assitant. Use only provided context to answer the question.
        If the answer is not preent in the context, say:"I could not find the answer in the question" """),
        ("human", """Context: {context}
                    Question: {question}""")
    ]
)

print("RAG system created")
print("Press 0 to exit")

while True:
    query = input("You:")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })
    
    response = llm.invoke(final_prompt)
    
    print(f"\n AI: {response.content}")