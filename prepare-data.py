import os
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
warnings.simplefilter(action='ignore', category=FutureWarning)


lecture_notes = """
Your lecture notes text here...
"""

segments = lecture_notes.split('\n\n')
model = SentenceTransformer('models/text-bison-001')

embeddings = model.encode(segments, convert_to_tensor=True)

os.makedirs('data', exist_ok=True)
with open('data/segments.json', 'w') as f:
    json.dump(segments, f)

torch.save(embeddings, 'data/embeddings.pt')

# Example LLM architectures table
data = {
    "Model": ["GPT-3", "BERT", "T5"],
    "Paper": ["Language Models are Few-Shot Learners", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"],
    "Year": [2020, 2019, 2020]
}

df = pd.DataFrame(data)

# Save table to JSON
df.to_json('data/llm_architectures.json', orient='records')

embeddings_np = embeddings.numpy()
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

faiss.write_index(index, 'data/faiss_index.index')