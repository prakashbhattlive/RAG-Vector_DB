import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load environment variables
load_dotenv()

# Local embedding configuration
OLLAMA_URL = "http://192.168.1.6:11434"
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

# Define prompt
prompt = PromptTemplate.from_template("Answer the following question:\n{question}")

if __name__ == "__main__":
    print("Retrieving ......")
    query = "What is Pinecone in machine learning?"
    chain = prompt | llm
    result = chain.invoke({"question": query})
    print("Result:", result)

    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embedding_model
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt=retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
    vectorstore.as_retriever(),
    combine_docs_chain
    )

    response = retrieval_chain.invoke({"input": query})
    print("Response:", response['answer'])