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
query = "Tell some news that happend on may 13 2025"
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

    


prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided context.
Use ALL the provided sources to answer the question.
If multiple perspectives exist, include them.
Don't start with half words.
Don't skip anything in the end.</s>
<|user|>
{query}

Context:
{context}</s>
<|assistant|>"""

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="modgeek/tinyllama-rag-finetuned",
    device=0  # 0 = GPU, -1 = CPU
)


# Load model directly
# from transformers import AutoModel
# model = AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct", dtype="auto")

llm = HuggingFacePipeline(pipeline=pipe)

print(llm.invoke(prompt))


# Load model directly
# from huggingface_hub import snapshot_download

# # Downloads entire model to local folder
# snapshot_download(
#     repo_id="modgeek/tinyllama-rag-finetuned",
#     local_dir="./tinyllama-rag-finetuned"
# )

# print("Model downloaded!")
