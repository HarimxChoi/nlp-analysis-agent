import torch
import sys
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


# 1. load data
df = pd.read_csv(r'./data/preprocessed_balanced_dataset.csv', encoding='cp949')

# 2. load SBERT model
model = SentenceTransformer('jhgan/ko-sbert-sts')

# 3. create sentence embeddings (GPU)
print("Creating sentence embeddings...")
embeddings = model.encode(df['service_title'].tolist(), show_progress_bar=True, convert_to_tensor=True, device='cuda')
embeddings = embeddings.cpu().numpy().astype('float32')  # Faiss uses numpy arrays

# 4. build Faiss index
print("Building Faiss index...")
index = faiss.IndexFlatL2(embeddings.shape[1])  # simple L2-distance index
index.add(embeddings)

# 5. save index and metadata
output_dir = r'./vector_db'
os.makedirs(output_dir, exist_ok=True)

faiss.write_index(index, os.path.join(output_dir, "bid_db.index"))

id_to_data = df[['service_title', 'label']].to_dict(orient='index')
with open(os.path.join(output_dir, "id_to_data.json"), 'w', encoding='utf-8') as f:
    json.dump(id_to_data, f, ensure_ascii=False, indent=4)

print("Vector DB built.")
