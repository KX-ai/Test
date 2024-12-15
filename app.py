import streamlit as st
from transformers import pipeline  # For GPT-NeoX integration (example with Hugging Face)
from PyPDF2 import PdfReader  # For PDF text extraction

# Initialize the app
st.title("Interactive Chatbot with Document Content")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or document", type=["pdf", "docx", "txt"])

# Function to extract text
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    # Add handling for other formats (e.g., .docx, .txt) here
    return None

# Load the GPT-NeoX model (example using Hugging Face Transformers)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neox-20b")  # Replace with your model setup

# Process file and queries
if uploaded_file:
    document_text = extract_text(uploaded_file)
    if document_text:
        st.text_area("Extracted Text", document_text[:5000], height=200)  # Display first 5000 characters

        # User query
        query = st.text_input("Ask a question:")
        if query:
            model = load_model()
            prompt = f"Document context: {document_text}\nUser question: {query}\nAnswer:"
            response = model(prompt, max_length=150, do_sample=True, temperature=0.7)
            st.write("Response:", response[0]['generated_text'])
    else:
        st.error("Failed to extract text. Please upload a valid document.")
