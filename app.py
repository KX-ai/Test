import streamlit as st
import openai
import PyPDF2  # Use PyPDF2 for PDF extraction

# Set OpenAI API key
openai.api_key = "73774827-17f1-49dc-bdc1-72362a5079a8"  # Replace with actual API key

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("Chatbot with PDF Content")
st.write("Upload a PDF file and ask questions about its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the uploaded PDF
    text_content = extract_text_from_pdf(pdf_file)
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
