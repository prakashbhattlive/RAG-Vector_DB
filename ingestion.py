import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OllamaEmbeddings


##local embedding configuration
OLLAMA_URL = "http://192.168.1.6:11434"
OLLAMA_MODEL = "mxbai-embed-large:latest"


# ✅ Initialize Ollama embedding model with remote base_url
embedding_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL  # Points to your VM
)

# Load environment variables
load_dotenv()

def validate_env_vars():
    required_vars = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_REGION"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

def load_and_split_documents(file_path, chunk_size=1000, chunk_overlap=200):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    print(f"📄 Loaded {len(documents)} documents, split into {len(chunks)} chunks.")
    return chunks

def main():
    print("🚀 Starting Pinecone ingestion with new SDK...")

    validate_env_vars()

    # Load env vars
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    region = os.getenv("PINECONE_REGION")

    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    print("✅ Pinecone client initialized.")

    # Create index if needed
    if index_name not in pc.list_indexes().names():
        print(f"🔧 Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=1024, # Match your embedding size
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": region}}
        )
        print(f"✅ Index '{index_name}' created.")
    else:
        print(f"📋 Using existing index '{index_name}'.")

    # Get index object
    index = pc.Index(index_name)

    # Load and split documents
    chunks = load_and_split_documents("vectordb.txt")

    # Initialize embeddings
    try:
        embeddings = embedding_model
        print("✅ local AIEmbeddings initialized.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to initialize embeddings: {e}")

    # Store vectors using LangChain's new PineconeVectorStore
    try:
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        vectorstore.add_documents(chunks)
        print(f"✅ Ingestion complete: {len(chunks)} chunks stored in Pinecone index '{index_name}'.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to store documents in Pinecone: {e}")

if __name__ == "__main__":
    main()