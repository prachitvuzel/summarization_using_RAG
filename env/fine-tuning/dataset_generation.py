import pickle
import random
import json
from tqdm import tqdm


with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f) 

#getting the the guardian key
index = 0
for i,chunk in enumerate(chunks): 
    url = chunk.metadata.get("url", "")
    if "wikipedia" not in url:
        print(chunk.metadata)
        index = i
        break





print(f"Total chunks: {len(chunks)}")
print(f"Sample chunk:\n{chunks[0]}")




RAG_TEMPLATES = [
    "Summarise the key points from the context.",
    "Give a brief overview of the context.",
    "In 2-3 sentences, what is the context about?",
    "What is the main idea being discussed?",
    "Provide a concise summary of the context.",
    "What is the passage mainly about?",
    "Condense the context into a few key points.",
    "What is the main topic discussed in the context?",
    "What does the context say about this topic?",
    "What are the most important details in the context?",
    "List the key facts mentioned in the context.",
    "What specific information is provided in the context?",
    "What facts can be confirmed from the context?",
    "What are the main points stated in the context?",
    "What conclusion can be drawn from the context?",
    "Based on the context, what happened and why?",
    "What can be inferred from the information provided?",
    "What does the context imply about this situation?",
    "Why did this happen according to the context?",
    "What is the significance of the information in the context?",
    "What outcome is suggested by the context?",
    "Extract any names, places, or dates from the context.",
    "List all the people or organisations mentioned in the context.",
    "What locations are mentioned in the context?",
    "Are there any statistics or numbers in the context? List them.",
    "What events are described in the context?",
    "Identify any key terms or concepts in the context.",
    "What time period does the context refer to?",
    "What is the main argument or claim made in the context?",
    "What point is being conveyed in the context?",
    "What is the tone of the context?",
    "What perspective is presented in the context?",
    "What stance is taken in the context?",
    "What evidence is provided in the context?",
    "What examples are given in the context?",
    "How does the context justify its claims?",
    "What reasoning is presented in the context?",
    "What details support the main idea in the context?",
    "Does the context provide enough information to answer this? If not, say so.",
    "Answer only from the context. If the answer isn't there, say 'Not found in context'.",
    "Is this claim supported by the context? Answer yes or no and explain.",
    "Only use the context to answer. Do not use outside knowledge.",
    "If the context does not contain the answer, explicitly state that.",
    "Are there any contrasting ideas presented in the context?",
    "What relationships between concepts are described in the context?",
    "How do the ideas in the context connect to each other?",
    "What cause and effect relationships are mentioned in the context?",
    "What event is being described in the context?",
    "Who are the key people involved according to the context?",
    "What happened according to the context?",
    "What are the implications of the event described in the context?",
    "What background information is provided about the event in the context?",
    "How is the main concept defined in the context?",
    "What historical background is provided in the context?",
    "What does the context tell us about the origin of this topic?",
    "How does the context describe the significance of this subject?",
    "What does the context reveal about the development of this topic?",
]




def build_dataset(docs, name, examples_per_chunk: int = 5):
    
    with open(f"finetuning_dataset_{name}.jsonl", "w") as f:
        for i in tqdm(range(len(docs) - 1), desc="Building dataset"):
            
            current_doc = docs[i]
            next_doc    = docs[i + 1]
            
            current_chunk = current_doc.page_content.strip()
            next_chunk    = next_doc.page_content.strip()
            

            if len(current_chunk) < 100 or len(next_chunk) < 100:
                continue

            if current_doc.metadata.get("title") != next_doc.metadata.get("title"):
                continue 
            
            questions = random.sample(RAG_TEMPLATES, examples_per_chunk)
            
            for question in questions:
                item = {
                    "instruction": question,
                    "input": current_chunk,
                    "output": next_chunk 
                }
                f.write(json.dumps(item) + "\n")

build_dataset(chunks[:index], "wikipedia", examples_per_chunk=5)
build_dataset(chunks[index:],  "the_guardian", examples_per_chunk=5)
