import streamlit as st
from transformers import pipeline  # For GPT-NeoX integration (example with Hugging Face)
from PyPDF2 import PdfReader  # For PDF text extraction

# Initialize the app
st.title("Interactive Chatbot with Document Content")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF or document", type=["pdf", "docx", "txt"])

# Function to extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages)
    # Add handling for other formats (e.g., .docx, .txt) here
    return None

# Load the GPT-NeoX model (using the Hugging Face Transformers pipeline)
def load_model():
    # Try loading with CUDA if available, else fall back to CPU
    return pipeline("text-generation", model="EleutherAI/gpt-neox-20b", device=0 if torch.cuda.is_available() else -1)

# Process file and queries
if uploaded_file:
    document_text = extract_text(uploaded_file)
    if document_text:
        # Show the extracted text (first 5000 characters)
        st.text_area("Extracted Text", document_text[:5000], height=200)

        # User query for the chatbot
        query = st.text_input("Ask a question:")
        if query:
            # Load model every time a query is asked to avoid caching issues
            model = load_model()
            prompt = f"Document context: {document_text}\nUser question: {query}\nAnswer:"
            try:
                # Generate response using the model
                response = model(prompt, max_length=150, do_sample=True, temperature=0.7)
                st.write("Response:", response[0]['generated_text'])
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.error("Failed to extract text. Please upload a valid document.")
