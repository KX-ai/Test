import streamlit as st
import openai
import fitz  # PyMuPDF for PDF extraction

# Set OpenAI API key
openai.api_key = "73774827-17f1-49dc-bdc1-72362a5079a8"  # Replace with your actual API key

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        # Open the PDF file with PyMuPDF
        with fitz.open("pdf", pdf_file.read()) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

# Streamlit UI
st.title("Chatbot with PDF Content")
st.write("Upload a PDF file and ask questions about its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the uploaded PDF
    text_content = extract_text_from_pdf(pdf_file)
    
    if text_content:
        st.write("PDF content extracted successfully.")
        
        # Store content for LLM interaction
        chat_history = []

        # Chat functionality
        user_input = st.text_input("Ask a question about the document:")

        if user_input:
            # Add user input to chat history
            chat_history.append(f"User: {user_input}")

            # Create prompt for Qwen 2.5 Instruct model using the extracted text (limit size)
            prompt = f"Document content: {text_content[:1000]}...\n\nUser question: {user_input}\nAnswer:"
            
            # Call the model to generate a response
            response = openai.Completion.create(
                model="Qwen-2.5-72B-Instruct",  # Replace with the actual model name
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7
            )

            # Get and display the response from the model
            answer = response.choices[0].text.strip()
            st.write(f"Qwen 2.5: {answer}")

            # Add model response to chat history
            chat_history.append(f"Qwen 2.5: {answer}")
