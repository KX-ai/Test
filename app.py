import streamlit as st
from transformers import pipeline  # Hugging Face pipeline
from PyPDF2 import PdfReader  # PDF text extraction
import torch

# Initialize the app
st.title("Interactive Chatbot with Document Content")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or document", type=["pdf", "docx", "txt"])

# Function to extract text from the uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    # Placeholder for handling other file formats (e.g., .docx, .txt)
    return None

# Load the GPT-Neo 1.3B model
@st.cache_resource
def load_model():
    try:
        # Use GPU if available, otherwise fallback to CPU
        device = 0 if torch.cuda.is_available() else -1
        return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=device)
    except Exception as e:
        st.error(f"Error loading GPT-Neo model: {e}")
        raise

# Process file and queries
if uploaded_file:
    document_text = extract_text(uploaded_file)
    if document_text:
        # Display the extracted text (first 5000 characters)
        st.text_area("Extracted Text", document_text[:5000], height=200)

        # Input for user query
        query = st.text_input("Ask a question:")
        if query:
            # Load model
            try:
                model = load_model()
                prompt = f"Document context: {document_text}\nUser question: {query}\nAnswer:"
                # Generate response
                response = model(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
                st.write("Response:", response[0]['generated_text'])
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.error("Failed to extract text. Please upload a valid document.")
else:
    st.info("Upload a document to get started.")
