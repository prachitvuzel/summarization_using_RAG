import faiss
import pickle
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from openai import OpenAI

index = faiss.read_index("vector_db.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("Index loaded:", index.ntotal)


#converting query into embedding

model = INSTRUCTOR('hkunlp/instructor-base')
instruction = "Represent the question for retrieving relevant documents"
query = "What is "
query_embedding = model.encode([[instruction, query]])[0]




query_vector = np.array([query_embedding]).astype("float32")
k = 5 
distances, indices = index.search(query_vector, k)


#retrieving chunks

retrieved_chunks = [chunks[i] for i in indices[0]]

for chunk in retrieved_chunks:
    print(chunk)
    print("----")


print("retrieved chunks:",retrieved_chunks)
#preparing to feed into LLMs

context = ""
for doc in retrieved_chunks:
        context += doc.page_content + "\n"

    


prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",
    device=0  # 0 = GPU, -1 = CPU
)

llm = HuggingFacePipeline(pipeline=pipe)

print(llm.invoke(prompt))
