import os
import openai
import PyPDF2
import streamlit as st

# Use the Sambanova API for Qwen 2.5-72B-Instruct
class SambanovaClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, model, messages, temperature=0.7, top_p=1.0):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p
        )

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

    # Initialize the Sambanova client with the API key
    api_key = os.environ.get("3a9006f7-a010-48b8-a17c-201155979015")  # Make sure to set your API key
    sambanova_client = SambanovaClient(
        api_key=api_key,
        base_url="https://api.sambanova.ai/v1"
    )

    # Store content for LLM interaction
    chat_history = [{"role": "system", "content": "You are a helpful assistant"}]

    # Chat functionality
    user_input = st.text_input("Ask a question about the document:")

    if user_input:
        # Add user input to chat history
        chat_history.append({"role": "user", "content": user_input})

        # Create prompt for Qwen 2.5 Instruct model using the extracted text (limit size)
        prompt = f"Document content: {text_content[:1000]}...\n\nUser question: {user_input}\nAnswer:"

        # Call the Qwen2.5-72B-Instruct model to generate a response
        response = sambanova_client.chat(
            model="Qwen2.5-72B-Instruct",  # Model name
            messages=chat_history,
            temperature=0.1,
            top_p=0.1
        )

        # Get and display the response from the model
        answer = response.choices[0].message['content'].strip()
        st.write(f"Qwen 2.5: {answer}")

        # Add model response to chat history
        chat_history.append({"role": "assistant", "content": answer})
