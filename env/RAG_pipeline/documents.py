import json
import os 
from langchain_core.documents import Document




wiki_science = []
wiki_history = []
wiki_mathematics = []
wiki_philosophy = []
wiki_linguistics = []
the_guardian_2025 = []
the_guardian_before = []

for file in os.listdir("."):

    if file.endswith(".json"):

        with open(file, "r", encoding="utf-8") as f:
            articles = json.load(f)

        # If JSON contains multiple articles
        for article in articles:

            doc = Document(
                page_content=article["text"],
                metadata={
                    "title": article["title"],
                    "url": article["url"]
                }
            )
            
            if "wikipedia_history" in file:
                wiki_history.append(doc)
            elif "wikipedia_science" in file:
                wiki_science.append(doc)
            elif "wikipedia_linguistics" in file:
                wiki_linguistics.append(doc)
            elif "wikipedia_mathematics" in file:
                wiki_mathematics.append(doc)
            elif "wikipedia_philosophy" in file:
                wiki_philosophy.append(doc)
            elif "theguardian_2025" in file:
                the_guardian_2025.append(doc)
            else:
                the_guardian_before.append(doc)

print("Documents loaded:", len(wiki_philosophy), len(wiki_history), len(wiki_mathematics) , len(wiki_science),len(wiki_linguistics), len(the_guardian_2025),len(the_guardian_before))