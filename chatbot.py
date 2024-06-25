from flask import Flask, request, render_template, jsonify
import google.generativeai as palm
palm.configure(api_key='AIzaSyD8C0qP40JCjPLdz0VM8Wk4Yy5AJ9rWPIM')
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('sata/faiss_index.index')

with open('sata/segments.json', 'r') as f:
    segments = json.load(f)

class ConversationalAgent:
    def _init_(self):
        self.context = []

    def query_index(self, query):
        query_embedding = model.encode([query], convert_to_tensor=True).numpy()
        _, indices = index.search(query_embedding, k=5)
        
        retrieved_segments = [segments[i] for i in indices[0]]
        return retrieved_segments

    def generate_response(self, query):
        retrieved_segments = self.query_index(query)
        self.context.extend(retrieved_segments)
        context = ' '.join(self.context[-10:])  # Use last 10 segments for context

        response = palm.generate_text(
        	model="models/text-bison-001",
        	prompt=prompt,
        	max_output_tokens=150
    	).result.strip()

agent = ConversationalAgent()

app = Flask(_name_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['query']
    response = agent.generate_response(user_query)
    return jsonify({'response': response})

if _name_ == '_main_':
    app.run(debug=True)
