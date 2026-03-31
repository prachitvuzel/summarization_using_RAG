import json

with open("theguardian_2025.json", "r", encoding="utf-8") as f:
                data = json.load(f)

print(len(data))