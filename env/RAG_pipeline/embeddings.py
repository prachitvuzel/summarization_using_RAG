from InstructorEmbedding import INSTRUCTOR
from chunking import all_chunks


model = INSTRUCTOR('hkunlp/instructor-base')
instruction = "Represent the information for retrieval"



batch_size = 256
embeddings = []

for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    texts = [[instruction, chunk.page_content] for chunk in batch]
    batch_embeddings = model.encode(texts)
    embeddings.extend(batch_embeddings)

print("Number of embeddings:",len(embeddings))