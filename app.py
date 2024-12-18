import streamlit as st
import openai
import fitz  # PyMuPDF

# Set OpenAI API key
openai.api_key = "96fbc73d8ea14f34ac9e179dc881c100"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI components
st.title("Chatbot with PDF Content")
st.write("Upload a PDF file and interact with its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the PDF
    text_content = extract_text_from_pdf(pdf_file)
    st.write("PDF content extracted successfully.")

    # Chat functionality
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        # Create a prompt for GPT-3.5 Turbo
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document content."},
            {"role": "user", "content": f"Document content: {text_content[:2000]}"},
            {"role": "user", "content": user_input}
        ]

        try:
            # Send the conversation to GPT-3.5 Turbo
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )

            # Display the response
            answer = response["choices"][0]["message"]["content"].strip()
            st.write(f"GPT-3.5: {answer}")

        except Exception as e:
            st.error(f"Error generating response: {e}")
