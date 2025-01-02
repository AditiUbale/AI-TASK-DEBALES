from langchain.document_loaders import URLLoader
import os

# URLs to scrape data from
urls = ["https://brainlox.com/courses/category/technical"]

# Initialize URLLoader
loader = URLLoader(urls=urls)

# Extract raw documents from URLs
documents = loader.load()

# Save extracted documents (you can save them as text or JSON)
os.makedirs("data/raw_documents", exist_ok=True)
with open("data/raw_documents/technical_courses.json", "w") as f:
    f.write(str(documents))
