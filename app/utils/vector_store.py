from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# load .env file  
load_dotenv()
COLLECTION_NAME = "tech_news"
PERSIST_DIRECTORY = "chromadb"

def get_vector_store():
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        ),
        persist_directory=PERSIST_DIRECTORY,
    )
    return vector_store


def add_to_vector_store(docs):
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
