from embeddings import embeddings
import faiss
import numpy as np


vectors = np.array(embeddings).astype("float32")
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
faiss.write_index(index, "vector_db.index")

print("vector database saved")