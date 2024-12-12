import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import io
from transformers import pipeline

# Function to load the GPT-Neo model using the pipeline
@st.cache_resource
def load_model():
    try:
        return pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
generator = load_model()

# Function to upload PDF file
def upload_file():
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.write(f"File uploaded: {uploaded_file.name}")
    return uploaded_file

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=io.BytesIO(pdf_file.read()), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to interact with GPT-Neo using the pipeline
def query_gpt_neox(text_prompt):
    if not generator:
        return "Model could not be loaded. Please try again later."
    try:
        outputs = generator(text_prompt, do_sample=True, min_length=50)
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# Function to handle user input and generate response based on document text
def chat_with_document(document_text, user_input):
    prompt = (
        f"Based on the following document text, answer the user's question:\n\n"
        f"Document Text: {document_text}\n\n"
        f"User Question: {user_input}\nAnswer:"
    )
    return query_gpt_neox(prompt)

# Main Streamlit app layout
def main():
    st.title("Interactive Chat with Document Content")

    # Check if the model loaded correctly
    if not generator:
        st.error("The GPT-Neo model could not be loaded. Please restart the app.")
        return

    # Upload PDF file
    uploaded_file = upload_file()

    if uploaded_file:
        # Extract text from the uploaded PDF
        document_text = extract_text_from_pdf(uploaded_file)
        if document_text:
            st.write("Document uploaded and text extracted successfully!")

            # Chat feature - ask a question about the document
            user_input = st.text_input("Ask a question about the document:")

            if user_input:
                # Generate response using GPT-Neo
                response = chat_with_document(document_text, user_input)
                st.write(f"Answer: {response}")
            else:
                st.write("Please enter a question to ask.")
        else:
            st.write("Failed to extract text from the document.")
    else:
        st.write("Please upload a PDF to start the conversation.")

if __name__ == "__main__":
    main()
