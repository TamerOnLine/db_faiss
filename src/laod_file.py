import os
import pandas as pd
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Define the FAISS database path
FAISS_PATH = "faiss_chat_index"

# Initialize embeddings
embeddings = OllamaEmbeddings(model="locusai/multi-qa-minilm-l6-cos-v1")

# Load FAISS database if it exists; otherwise, create a new one
if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
    faiss_index = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    print("✅ FAISS database loaded successfully.")
else:
    print("⚠️ FAISS database not found. Creating a new one...")
    faiss_index = FAISS.from_texts(["Welcome to the chatbot!"], embeddings)
    faiss_index.save_local(FAISS_PATH)
    print("✅ New FAISS database created and saved.")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    texts = [page.get_text("text") for page in doc]
    return [text.strip() for text in texts if text.strip()]

def dynamic_file_loader():
    """
    Dynamically loads data from a file selected by the user.
    
    The function prompts the user to enter a file path. If the file exists
    and is in a supported format (CSV, TXT, or PDF), it extracts textual data
    and adds it to the FAISS database.
    """
    file_path = input("📂 Enter the file path to load: ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found! Please check the path.")
        return

    texts = []
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        texts = df.iloc[:, 0].dropna().tolist()
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            texts = [line.strip() for line in file.readlines() if line.strip()]
    elif file_path.endswith(".pdf"):
        texts = extract_text_from_pdf(file_path)
    else:
        print("⚠️ Unsupported file format! Please use CSV, TXT, or PDF.")
        return

    if not texts:
        print("⚠️ No text extracted from the file.")
        return

    print(f"📥 Loading {len(texts)} entries into the FAISS database...")
    faiss_index.add_texts(texts)
    faiss_index.save_local(FAISS_PATH)
    print("✅ FAISS database updated successfully!")

# Run the dynamic file loader if the script is executed directly
if __name__ == "__main__":
    dynamic_file_loader()
