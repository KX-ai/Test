import streamlit as st
import openai
import fitz  # PyMuPDF for PDF extraction
import io  # for handling BytesIO

# Set OpenAI API key
openai.api_key = "73774827-17f1-49dc-bdc1-72362a5079a8"  # Replace with your actual API key

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    # Open the PDF file from BytesIO object
    pdf_file_bytes = io.BytesIO(pdf_file.read())
    doc = fitz.open(pdf_file_bytes)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit UI
st.title("Chatbot with PDF Content")
st.write("Upload a PDF file and ask questions about its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the uploaded PDF
    try:
        text_content = extract_text_from_pdf(pdf_file)
        st.write("PDF content extracted successfully.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        text_content = ""

    if text_content:
        # Store content for LLM interaction
        chat_history = []

        # Chat functionality
        user_input = st.text_input("Ask a question about the document:")

        if user_input:
            # Add user input to chat history
            chat_history.append(f"User: {user_input}")

            # Create prompt for Qwen 2.5 Instruct model using the extracted text (limit size)
            prompt = f"Document content: {text_content[:1000]}...\n\nUser question: {user_input}\nAnswer:"
            
            try:
                # Call the model to generate a response
                response = openai.ChatCompletion.create(
                    model="Qwen-2.5-72B-Instruct",  # Replace with the actual model name if different
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.7
                )

                # Get and display the response from the model
                answer = response.choices[0].message['content'].strip()
                st.write(f"Qwen 2.5: {answer}")

                # Add model response to chat history
                chat_history.append(f"Qwen 2.5: {answer}")
            except Exception as e:
                st.error(f"Error getting response from model: {e}")
