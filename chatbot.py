import os
import faiss
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from flask import Flask, request, jsonify
import openai
from langchain_community.llms import OpenAI
import langchain

# Ensure you have your OpenAI API key set
openai.api_key = "your-openai-api-key"

# Create a Flask app
app = Flask(__name__)

# Load your documents from the web
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
docs = loader.load()

# Create embeddings and store them in a vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

@app.route("/ask", methods=["POST"])
def ask():
    # Get question from the API request
    user_input = request.json.get("question")
    
    # Retrieve the most relevant documents based on the question
    docs = vectorstore.similarity_search(user_input, k=1)  # Adjust 'k' for number of results
    
    # Get answer from OpenAI model using retrieved docs
    llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=openai.api_key)
    response = llm.call(docs)
    
    # Return response as JSON
    return jsonify({"response": response['text']})


if __name__ == "__main__":
    app.run(debug=True)
