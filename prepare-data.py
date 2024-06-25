import os
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


lecture_notes = "../Data/lecture notes.txt"

segments = lecture_notes.split('\n\n')
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(segments, convert_to_tensor=True)

os.makedirs('sata', exist_ok=True)
with open('sata/segments.json', 'w') as f:
    json.dump(segments, f)

torch.save(embeddings, 'sata/embeddings.pt')

data = {"../LLM Architecture/CodeLlama-7B-Instruct.json",
        "../LLM Architecture/Hermes-2-Pro-Mistral-7B.json",
        "../LLM Architecture/Llama-3-8B-Instruct.json",
        "../LLM Architecture/Mistral-7B-Instruct-v0.1.json",
        "../LLM Architecture/Mistral-7B-Instruct-v0.2.json",
        "../LLM Architecture/OpenHermes-2.5-Mistral-7B.json",
        "../LLM Architecture/deepseek-coder-6.7b-instruct.json",
        "../LLM Architecture/google-gemma-2b.json",
        "../LLM Architecture/phi-2.json"
}

df = pd.DataFrame(data)

df.to_json('sata/llm_architectures.json', orient='records')

embeddings_np = embeddings.numpy()
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

faiss.write_index(index, 'sata/faiss_index.index')
