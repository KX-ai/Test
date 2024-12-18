import streamlit as st
import openai
import fitz  # PyMuPDF

# Set OpenAI API key
openai.api_key = "47d1a9054d674edb96b4db6ea1c47ff2"  # Replace with your actual key

# Create a function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        # Open the uploaded file as a stream
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

# Streamlit UI components
st.title("Chatbot with PDF Content")
st.write("Upload a PDF file and interact with its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    try:
        # Extract text from the PDF
        text_content = extract_text_from_pdf(pdf_file)
        st.write("PDF content extracted successfully.")
        
        # Store content for LLM interaction
        chat_history = []

        # Chat functionality
        user_input = st.text_input("Ask a question about the document:")

        if user_input:
            # Add user input to chat history
            chat_history.append(f"User: {user_input}")

            # Create prompt for GPT-3.5 model using the PDF content
            prompt = f"Document content: {text_content[:1000]}...\n\nUser question: {user_input}\nAnswer:"
            
            # Send the prompt to GPT-3.5
            try:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo",
                    prompt=prompt,
                    max_tokens=150,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
                # Get and display the model's response
                answer = response.choices[0].text.strip()
                st.write(f"GPT-3.5: {answer}")

                # Add model response to chat history
                chat_history.append(f"GPT-3.5: {answer}")
            except Exception as e:
                st.error(f"Failed to generate a response from GPT-3.5: {e}")

    except RuntimeError as e:
        st.error(e)
else:
    st.info("Please upload a PDF file.")
