import re
import json
import sys


        

def clean_text(text):
    text = re.sub(r"\[+[^\[\]]*\]+", "", text)  
    text = re.sub(r"\[\s*\d+\s*\]", "", text) 
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# for file in sys.argv[1:]:
#     print(file.split("."))
for file in sys.argv[1:]:
        with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cleaned_articles = []


        for article in data:
            cleaned_text = clean_text(article["content"])
            cleaned_articles.append({
                "title": article["title"],
                "url": article["url"],
                "text": cleaned_text
            })

        with open(file.split(".")[0]+"_cleaned.json", "w", encoding="utf-8") as f:
                json.dump(cleaned_articles, f, ensure_ascii=False, indent=2)

        






