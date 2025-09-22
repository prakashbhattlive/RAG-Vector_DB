import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Local embedding configuration
OLLAMA_URL = "http://192.168.1.6:11434" # Replace with your local Ollama server URL or localhost if running locally in a same machine
OLLAMA_MODEL = "mxbai-embed-large:latest"

# Initialize embedding model
embedding_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

# Initialize LLM
llm = Ollama(
    model="llama3:latest",
    base_url=OLLAMA_URL
)

if __name__ == "__main__":
    pdf_path = "Vector_DB/Vectors_in memory/react.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)
    #print(f"Loaded {len(docs)} documents from {pdf_path}")

    embeddings = embedding_model
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    #print("Vector store saved to 'faiss_index_react' directory")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    
    retrirval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt=retrirval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),
        combine_docs_chain
    )

    response = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(response['answer'])