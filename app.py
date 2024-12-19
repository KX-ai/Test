import os
import openai
import PyPDF2
import streamlit as st

# Use the Sambanova API for Qwen 2.5-72B-Instruct
class SambanovaClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        openai.api_key = self.api_key  # Set the API key for the OpenAI client
        openai.api_base = self.base_url  # Set the base URL for the OpenAI API

    def chat(self, model, messages, temperature=0.7, top_p=1.0, max_tokens=500):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p
            )
            return response
        except Exception as e:
            raise Exception(f"Error while calling OpenAI API: {str(e)}")

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.title("Chatbot with PDF Content (Botify)")
st.write("Upload a PDF file and ask questions about its content.")

# File upload
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the uploaded PDF
    text_content = extract_text_from_pdf(pdf_file)
    st.write("PDF content extracted successfully.")

    # Retrieve the API key securely from Streamlit Secrets
    api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
    if not api_key:
        st.error("API key not found! Please check your secrets settings.")
    else:
        sambanova_client = SambanovaClient(
            api_key=api_key,
            base_url="https://api.sambanova.ai/v1"
        )

        # Initialize session state to store chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": "You are a helpful assistant named Botify."}]

        # Truncate chat history to the last 5 exchanges to avoid token limit issues
        max_history = 5
        st.session_state.chat_history = st.session_state.chat_history[-max_history:]

        # Temporary variable for user input to avoid modifying session_state directly
        temp_user_input = st.text_input("Ask a question about the document:", key="user_input")

        if temp_user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": temp_user_input})

            # Truncate document content to fit within token limits
            max_content_length = 500  # Adjust as needed
            truncated_content = text_content[:max_content_length]

            # Create prompt for the model
            prompt_text = f"Document content (truncated): {truncated_content}...\n\nUser question: {temp_user_input}\nAnswer:"
            st.session_state.chat_history.append({"role": "system", "content": prompt_text})

            try:
                # Call the Qwen2.5-72B-Instruct model to generate a response
                response = sambanova_client.chat(
                    model="Qwen2.5-72B-Instruct",
                    messages=st.session_state.chat_history,
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=300  # Reduce output size to fit within token limits
                )

                # Extract and display the response
                answer = response['choices'][0]['message']['content'].strip()
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.write(f"Botify: {answer}")

                # Clear the temporary input (reset functionality)
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error occurred while fetching response: {str(e)}")

        # Display the chat history
        with st.expander("Chat History"):
            for msg in st.session_state.chat_history:
                role = "User" if msg["role"] == "user" else "Botify"
                st.write(f"**{role}:** {msg['content']}")
