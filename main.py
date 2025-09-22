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
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

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


    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep it concise and to the point.
    Always say "thanks for asking!" at the end of your answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()} 
            | custom_rag_prompt 
            | llm 
    )

    rag_response = rag_chain.invoke({"question": query})
    print(rag_response)





