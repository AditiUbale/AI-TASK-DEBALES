from flask import Flask, request, jsonify
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Retrieve OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")

# Initialize OpenAI embeddings with your API key
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load the stored FAISS vector store from the disk
vectorstore_path = "models/faiss_index"
vectorstore = FAISS.load_local(vectorstore_path, embeddings)

@app.route('/query', methods=['POST'])
def query_chatbot():
    try:
        # Get the user's question from the request
        user_query = request.json.get('query')

        # Perform the search in the vector store
        results = vectorstore.similarity_search(user_query, k=3)

        # Format response data with top results
        response_data = [
            {
                "content": result['content'],
                "metadata": result['metadata']
            }
            for result in results
        ]

        # Return the response as JSON
        return jsonify({"response": response_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
