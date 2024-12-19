import os
import openai
import PyPDF2
import streamlit as st
import time

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
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit UI
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Chatbot with PDF Content (Botify)")

# Upload a PDF
st.write("Upload a PDF file and interact with the chatbot to ask questions.")
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    # Extract text from the uploaded PDF
    text_content = extract_text_from_pdf(pdf_file)
    st.success("PDF content extracted successfully!")

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

        # Display the chat conversation
        st.write("### Chat Conversation")
        chat_container = st.container()

        # Display chat history dynamically
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**🧑 User:** {msg['content']}")
                elif msg["role"] == "assistant":
                    st.markdown(f"**🤖 Botify:** {msg['content']}")

        # User input at the bottom
        user_input = st.text_input(
            "Your message:", 
            key="user_input", 
            placeholder="Type your message and press Enter to send..."
        )

        if user_input:
            # Add user input to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Truncate document content to fit within token limits
            max_content_length = 500  # Optimized for performance
            truncated_content = text_content[:max_content_length]

            # Create prompt for the model
            prompt_text = f"Document content (truncated): {truncated_content}...\n\nUser question: {user_input}\nAnswer:"
            st.session_state.chat_history.append({"role": "system", "content": prompt_text})

            # Measure API call time
            start_time = time.time()
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

                # Refresh the chat container to display the latest interaction
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error occurred while fetching response: {str(e)}")
            finally:
                end_time = time.time()
                st.info(f"API call duration: {end_time - start_time:.2f} seconds")
