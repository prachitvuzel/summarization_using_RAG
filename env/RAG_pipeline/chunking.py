from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents import wiki_science, wiki_history, wiki_philosophy, wiki_linguistics, wiki_mathematics, the_guardian_2025, the_guardian_before
import pickle

# Initialize splitter (you can tweak chunk size per source)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Chunk each category
wiki_science_chunks = splitter.split_documents(wiki_science)
wiki_philosophy_chunks = splitter.split_documents(wiki_philosophy)
wiki_history_chunks = splitter.split_documents(wiki_history)
wiki_linguistics_chunks = splitter.split_documents(wiki_linguistics)
wiki_mathematics_chunks = splitter.split_documents(wiki_mathematics)
guardian_2025_chunks = splitter.split_documents(the_guardian_2025)
guardian_before_chunks = splitter.split_documents(the_guardian_before)




wikipedia_chunks = ( wiki_science_chunks +
    wiki_philosophy_chunks +
    wiki_history_chunks +
    wiki_mathematics_chunks+
    wiki_linguistics_chunks)

the_guardian_chunks = (  guardian_2025_chunks +
    guardian_before_chunks)

# all_chunks = (
#     wiki_science_chunks +
#     wiki_philosophy_chunks +
#     wiki_history_chunks +
#     wiki_mathematics_chunks+
#     wiki_linguistics_chunks+
#     guardian_2025_chunks +
#     guardian_before_chunks
# )

print("Total chunks ready for embeddings:", len(wikipedia_chunks)+len(the_guardian_chunks))

with open("chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("Chunks saved")