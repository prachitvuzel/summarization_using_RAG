import faiss
import pickle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from InstructorEmbedding import INSTRUCTOR

# ── 2. Device Setup ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

# ── 3. Load Vector DB ─────────────────────────────────────────────
print("\nLoading vector DB...")
index = faiss.read_index("vector_db.index")
print(f"Index loaded : {index.ntotal} vectors")

# ── 4. Load Chunks ────────────────────────────────────────────────
print("\nLoading chunks...")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Chunks loaded : {len(chunks)}")

# ── 5. Load Embedding Model ───────────────────────────────────────
print("\nLoading embedding model...")
embedding_model = INSTRUCTOR("hkunlp/instructor-base")
print("Embedding model loaded!")

# ── 6. Load Qwen2.5 1.5B ─────────────────────────────────────────
print("\nLoading Qwen2.5 1.5B...")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

# ── 7. Load Tokenizer ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ── 8. Set Model to Inference Mode ───────────────────────────────
model.eval()

print(f"Model loaded!")
print(f"Model device : {next(model.parameters()).device}")

if torch.cuda.is_available():
    print(f"Memory used  : {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")
else:
    print("Running on CPU")

# ── 9. Retrieval Function ─────────────────────────────────────────
def retrieve_chunks(query: str, k: int = 3) -> list:
    """
    Takes a question and returns k most relevant chunks
    from the vector database
    """

    # Embed the query using the same instructor model
    # that was used to build the vector database
    instruction     = "Represent the question for retrieving relevant documents"
    query_embedding = embedding_model.encode([[instruction, query]])[0]

    # Convert to float32 array — FAISS requires this format
    query_vector = np.array([query_embedding]).astype("float32")

    # Search the vector database for k nearest chunks
    distances, indices = index.search(query_vector, k)

    # Return the actual chunk objects
    retrieved = [chunks[i] for i in indices[0]]

    return retrieved

# ── 10. Answer Generation Function ───────────────────────────────
def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Takes a question and retrieved chunks
    and generates a complete answer using Qwen
    """

    # Build numbered context from all retrieved chunks
    # so the model knows which source each piece comes from
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"Source {i + 1}:\n{chunk.page_content}\n\n"

    # Qwen uses a messages format — system + user roles
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that answers questions using ONLY the provided sources.
Synthesize information from ALL sources given.
Give a complete and coherent answer.
Do not cut off mid sentence.
If the sources do not contain enough information say so clearly."""
        },
        {
            "role": "user",
            "content": f"""Question: {question}

{context}
Based on ALL the sources above provide a complete answer to the question."""
        }
    ]

    # Apply Qwen's built in chat template
    # This formats the messages correctly for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the formatted text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=3072,          # Qwen supports long context ✅
    ).to(device)

    # Generate the answer
    with torch.no_grad():         # no gradient calculation needed for inference
        output = model.generate(
            **inputs,
            max_new_tokens=500,   # enough for a complete answer
            temperature=0.3,      # low temperature = more focused answers
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # prevents repeating the same sentences
        )

    # Decode only the newly generated tokens
    # not the input prompt — avoids prompt bleed into answer
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer

# ── 11. Full RAG Pipeline Function ───────────────────────────────
def rag_query(question: str, k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from vector DB
    2. Generate answer using Qwen
    3. Return answer with sources
    """

    print(f"\nQuestion  : {question}")
    print("Retrieving relevant chunks...")

    # Step 1 — Retrieve
    retrieved = retrieve_chunks(question, k=k)
    print(f"Retrieved  : {len(retrieved)} chunks")

    # Preview retrieved sources
    for i, chunk in enumerate(retrieved):
        print(f"  Source {i+1} : {chunk.metadata.get('url', 'unknown')[:60]}")

    print("Generating answer...")

    # Step 2 — Generate
    answer  = generate_answer(question, retrieved)
    sources = [chunk.metadata.get("url", "") for chunk in retrieved]

    return {
        "question" : question,
        "answer"   : answer,
        "sources"  : sources
    }

# ── 12. Test It ───────────────────────────────────────────────────
result = rag_query("tell me some exciting news on 2025 february")

print(f"\n── Result ───────────────────────────────────────────")
print(f"Question : {result['question']}")
print(f"\nAnswer   :\n{result['answer']}")
print(f"\nSources  :")
for source in result['sources']:
    print(f"  → {source}")
print("─────────────────────────────────────────────────────")