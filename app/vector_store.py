from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# load .env file  
load_dotenv()


def get_vector_store():
    vector_store = Chroma(
        collection_name="tech_news",
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        ),
        persist_directory="./chromadb"
    )
    return vector_store


def add_to_vector_store(docs):
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
