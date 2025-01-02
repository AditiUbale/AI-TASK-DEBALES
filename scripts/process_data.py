import json

# Load raw documents
with open("data/raw_documents/technical_courses.json", "r") as f:
    documents = json.load(f)

# Process documents (e.g., clean HTML, remove unnecessary tags)
processed_documents = []
for doc in documents:
    processed_documents.append(doc['content'])  # Example: Extract content

# Save processed documents
with open("data/processed_documents/cleaned_courses.txt", "w") as f:
    f.write("\n".join(processed_documents))
