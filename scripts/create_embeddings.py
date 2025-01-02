from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Load processed documents
with open("data/processed_documents/cleaned_courses.txt", "r") as f:
    documents = f.readlines()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key_here")

# Create a FAISS vector store and save embeddings
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("models/faiss_index")
