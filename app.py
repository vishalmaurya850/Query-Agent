import google.generativeai as palm
palm.configure(api_key=APIKEY)
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import faiss
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
with open('sata/segments.json', 'r') as f:
    segments = json.load(f)
class QueryAgent:
    def __init__(self):
        self.index = None
        self.data_segments = []
        self.load_index()
        self.load_model()
        self.context = []  # Initialize context as an empty list

    def load_index(self):
        # Load the index using faiss.read_index
        self.index = faiss.read_index('sata/faiss_index.index')

    def load_model(self):
        # Load the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_query(self, query):
        # Generate embeddings using the sentence transformer model
        query_embedding = self.model.encode(query)
        return query_embedding

    def query_index(self, query):
        # Ensure query_embedding is in the correct shape
        query_embedding = self.model.encode([query], convert_to_tensor=True).numpy()
        _, indices = self.index.search(query_embedding, k=5)
        retrieved_segments = [segments[i] for i in indices[0]]
        return retrieved_segments

    def generate_response(self, query):
        # Retrieve the actual data segments using the indices
        retrieved_segments = self.query_index(query)
        self.context.extend(retrieved_segments)
        # Join the last 10 entries in the context into a single string
        context= ' '.join(map(str, self.context[-10:]))
        # Process retrieved_segments to generate a response
        prompt = (f"Answer the following question based on the context provided:\n\n"
                  f"Context:{context}\n\nQuestion: {query}\n\nAnswer:")

        print("Prompt for Gemini API:", prompt) 

        response = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            max_output_tokens=200
        )
        return response.result.strip()
    

# Initialize the QueryAgent
agent = QueryAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']
    response = agent.generate_response(user_query)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
