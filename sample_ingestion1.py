import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OllamaEmbeddings

# Load env vars
OLLAMA_URL = "http://192.168.1.6:11434"
OLLAMA_MODEL = "mxbai-embed-large:latest"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME", "my-index")
region = os.getenv("PINECONE_REGION", "us-east-1")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Match your embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=region),
    )

# Get Pinecone Index
index = pc.Index(index_name)

# ✅ Initialize Ollama embedding model with remote base_url
embedding_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL  # Points to your VM
)

# Create LangChain-compatible Pinecone vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="text",  # required for storing and searching
)

# Add texts to vectorstore
texts = ["Hello world", "AI is amazing"]
vectorstore.add_texts(texts)

print("✅ Documents embedded and added to Pinecone.")