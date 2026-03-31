import pickle
import random
import json
from tqdm import tqdm


with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)  

print(f"Total chunks: {len(chunks)}")
# print(f"Sample chunk:\n{chunks[0:4]}")


content = ""
for chunk in chunks:
    url = chunk.metadata.get("url", "")
    if "wikipedia" not in url:
        print(chunk.metadata)
        break

print(content)
    