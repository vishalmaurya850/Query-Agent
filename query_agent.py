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
            prompt=f"Answer the following question based on the context provided:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
            max_output_tokens=200
        )
        return response.result.strip()

    def generate_response_with_citations(query, retrieved_segments):
        retrieved_segments = self.query_index(query)
        context = " ".join(retrieved_segments)
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:",
        response = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            max_output_tokens=150
        ).result.strip()
        citations = "\n".join([f"Citation: {segments}" for segments in retrieved_segments])

agent = ConversationalAgent()

def interact_with_user():
    print("You can start asking questions. Type 'exit' to end the session.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break
        response = agent.generate_response(user_query)
        print(f"Agent: {response}")

if __name__ == "__main__":
    interact_with_user()
