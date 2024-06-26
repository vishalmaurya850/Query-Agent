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
    def __init__(self):
        self.context = []

    def query_index(self, query):
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().numpy()

        _, indices = index.search(query_embedding, k=5)
        
        retrieved_segments = [segments[i] for i in indices[0]]
        return retrieved_segments

    def generate_response(self, query):
        retrieved_segments = self.query_index(query)
        print("Retrieved Segments:", retrieved_segments)  
        self.context.extend(retrieved_segments)
        context = ' '.join(self.context[-10:])

        prompt = (f"Answer the following question based on the context provided:\n\n"
                  f"Context: {context}\n\nQuestion: {query}\n\nAnswer:")

        print("Prompt for Gemini API:", prompt) 

        response = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            max_output_tokens=200
        )
        return response.result.strip()
        
        if response.result is not None:
            return response.result.strip()
        else:
            return "Sorry! I didn't Understand. I will make it correct."

agent = ConversationalAgent()

app = Flask(__name__)   

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
