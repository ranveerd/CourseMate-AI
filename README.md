# 📚 Book RAG Assistant

A simple **Retrieval-Augmented Generation (RAG)** application that lets you upload a PDF book and ask questions about its content.

Instead of sending the entire book to the LLM, the application first finds the most relevant parts of the uploaded book and then uses **Google Gemini** to generate an answer based on that content.

I built this project to understand how a practical RAG pipeline works, from loading a document and creating embeddings to retrieving relevant information and generating a final response.

---

## 🚀 What does this project do?

The application allows you to:

- 📄 Upload a PDF book
- ✂️ Split the book into smaller text chunks
- 🧠 Convert the chunks into vector embeddings
- 🗄️ Store the embeddings in ChromaDB
- 🔎 Retrieve relevant content when a question is asked
- 🤖 Generate a natural-language answer using Gemini
- 💻 Interact with everything through a simple Streamlit UI

The application focuses on answering questions **from the uploaded book** rather than relying on unrelated information.

---

## 🔄 How the RAG Pipeline Works

```text
                📄 PDF Book
                     │
                     ▼
              PyPDFLoader
                     │
                     ▼
             Text Splitting
                     │
                     ▼
       Hugging Face Embeddings
                     │
                     ▼
                ChromaDB
                     │
                     ▼
                Retriever
                     │
              User Question
                     │
                     ▼
              Relevant Chunks
                     │
                     ▼
              Google Gemini
                     │
                     ▼
              🤖 Final Answer
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangChain | Building the RAG pipeline |
| PyPDF | Loading PDF documents |
| RecursiveCharacterTextSplitter | Splitting documents into chunks |
| Hugging Face | Creating local text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| Google Gemini | Answer generation |
| Streamlit | Web interface |
| python-dotenv | Managing environment variables |

---

## 📂 Project Structure

```text
RAG Project/
│
├── Document Loaders/
│   └── deeplearning.pdf
│
├── create_database.py
├── main.py
├── .env
├── .gitignore
└── README.md
```

> `chroma-db` is generated when the application creates the vector database and is not required to be manually created beforehand.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Move into the project folder

```bash
cd "RAG Project"
```

### 3. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file yet, install the required packages with:

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install langchain-google-genai
pip install langchain-huggingface
pip install chromadb
pip install sentence-transformers
pip install pypdf
pip install langchain-text-splitters
pip install python-dotenv
```

---

## 🔑 API Key Setup

This project uses **Google Gemini** for generating answers.

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep your API key private.

Do **not** upload your `.env` file to GitHub.

Add this to your `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
chroma-db/
```

---

## ▶️ Running the Application

Start the Streamlit application with:

```bash
streamlit run main.py
```

The application will open in your browser.

### Using the application

1. Upload a PDF book.
2. Wait for the document to be processed.
3. Enter a question related to the book.
4. The RAG system retrieves relevant sections.
5. Gemini generates the final answer from the retrieved content.

For example:

```text
Upload: Deep Learning.pdf

Question:
What is the Word2Vec framework?

Answer:
Word2Vec is a framework for generating word embeddings...
```

---

## 🧠 Why Hugging Face Embeddings?

Initially, the project was designed around API-based embeddings. However, using a local Hugging Face embedding model makes the vector creation process independent of a paid embedding API.

The project uses:

```text
all-MiniLM-L6-v2
```

This model converts text into numerical vectors that can be stored and searched in ChromaDB.

The advantage is that **embedding generation runs locally**, while Gemini is used for the final answer generation.

---

## 🗄️ Why ChromaDB?

ChromaDB is used as the vector database for this project.

When the PDF is processed:

```text
PDF
 ↓
Text Chunks
 ↓
Embeddings
 ↓
ChromaDB
```

When the user asks a question:

```text
Question
 ↓
Similarity Search
 ↓
Relevant Chunks
 ↓
Gemini
 ↓
Answer
```

This allows the application to work with information contained in the uploaded document without putting the complete document into every prompt.

---

## 🔍 Retrieval Strategy

The project uses **MMR (Maximum Marginal Relevance)** retrieval.

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

MMR helps retrieve relevant information while also reducing unnecessary similarity between the retrieved chunks.

---

## 📌 Current Scope

This project intentionally keeps the interface simple.

Currently, it supports:

- One uploaded PDF at a time
- Question answering based on the uploaded PDF
- Automatic creation of the vector database
- Gemini-powered responses
- Local Hugging Face embeddings

When a new PDF is uploaded, the previous Chroma database is replaced so that answers are based on the **currently uploaded book**.

---

## 🎯 What I Learned

While building this project, I got hands-on experience with:

- Document loading
- Text chunking
- Embeddings
- Vector databases
- Semantic search
- Retrieval-Augmented Generation
- LangChain
- ChromaDB
- Hugging Face embedding models
- Gemini LLM integration
- Building an AI application with Streamlit

More importantly, this project helped me understand how the different components of a RAG application connect together instead of just using an LLM directly.

---

## 🔮 Future Improvements

Some features I may explore in the future:

- 💬 Chat history
- 📚 Multiple PDF support
- 📑 Source/page references
- ⚡ Faster document processing
- 🎨 Improved Streamlit interface
- 🔎 Better retrieval and reranking
- ☁️ Deployment

---

## 👨‍💻 About the Project

This project was built as part of my journey toward becoming an **AI Engineer**, with a focus on understanding practical **Generative AI, LLM, and RAG applications**.

I'm continuously learning, experimenting, and building projects to strengthen my understanding of AI engineering.

---

## ⭐ If you found this project useful

Feel free to explore the code, experiment with the RAG pipeline, and build your own version.

**Built with Python 🐍 + LangChain 🔗 + ChromaDB 🗄️ + Gemini 🤖 + Streamlit 🎈**