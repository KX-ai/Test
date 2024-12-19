import os
import openai
import requests
import PyPDF2
import streamlit as st
import time
import json

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

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


# Use the Together API for Wizard LM-2 (8x22b)
class TogetherClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.together.xyz/v1/chat/completions"

    def chat(self, model, messages):
        payload = {
            "model": model,
            "messages": messages
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            return response.json()
        except Exception as e:
            raise Exception(f"Error while calling Together API: {str(e)}")


# Function to extract text from PDF using PyPDF2
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to load chat history from a JSON file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []  # Return an empty list if no history exists


# Function to save chat history to a JSON file
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file)


# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Chatbot with PDF Content (Botify)")

# Upload a PDF file
st.write("Upload a PDF file and interact with the chatbot to ask questions.")
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Initialize session state to store chat history and current chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. What can I help you today?"}]

# Button to start a new chat
if st.button("Start New Chat"):
    # Clear the current chat to start fresh
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. What can I help you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    st.success("New chat started! Feel free to ask your question.")

# Display the real-time chat conversation view
st.write("### Chat Conversation")

# Placeholder for the conversation, to be updated dynamically
conversation_placeholder = st.empty()

# Display the conversation dynamically
with conversation_placeholder.container():
    for msg in st.session_state.current_chat:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:  # Check if msg is a valid dictionary
            if msg["role"] == "user":
                st.markdown(f"**🧑 User:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**🤖 Botify:** {msg['content']}")
        else:
            st.error("Error: A message is missing or malformed in the current chat history.")

# API keys
sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
together_api_key = "db476cc81d29116da9b75433badfe89666552a25d2cd8efd6cb5a0c916eb8f50"

# Model selection
model_choice = st.selectbox("Select the LLM model:", ["Sambanova (Qwen 2.5-72B-Instruct)", "Together (Wizard LM-2 8x22b)"])

# Add prompt for user input when changing model
if "previous_model_choice" in st.session_state and st.session_state.previous_model_choice != model_choice:
    st.session_state.current_chat.append({"role": "assistant", "content": "Please type your message to continue."})

st.session_state.previous_model_choice = model_choice  # Store the selected model

# Wait for user input
user_input = st.text_input(
    "Your message:", 
    key="user_input", 
    placeholder="Type your message here and press Enter"
)

# Handle user input and send message
if user_input:
    # Add user input to current chat
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    # Handle PDF file and extract text
    if pdf_file is not None:
        # Extract text from the uploaded PDF
        text_content = extract_text_from_pdf(pdf_file)
        st.success("PDF content extracted successfully!")

        # Truncate document content to fit within token limits
        max_content_length = 500  # Optimized for performance
        truncated_content = text_content[:max_content_length]

        # Create prompt for the model
        prompt_text = f"Document content (truncated): {truncated_content}...\n\nUser question: {user_input}\nAnswer:"
        st.session_state.current_chat.append({"role": "system", "content": prompt_text})

        # Measure API call time
        start_time = time.time()
        try:
            if model_choice == "Sambanova (Qwen 2.5-72B-Instruct)":
                # Call the Qwen2.5-72B-Instruct model to generate a response
                response = SambanovaClient(
                    api_key=sambanova_api_key,
                    base_url="https://api.sambanova.ai/v1"
                ).chat(
                    model="Qwen2.5-72B-Instruct",
                    messages=st.session_state.current_chat,
                    temperature=0.1,
                    top_p=0.1,
                    max_tokens=300  # Reduce output size to fit within token limits
                )

                # Extract the answer from the Sambanova response
                answer = response['choices'][0]['message']['content'].strip()

            elif model_choice == "Together (Wizard LM-2 8x22b)":
                # Call the Wizard LM-2 (8x22b) model to generate a response
                response = TogetherClient(api_key=together_api_key).chat(
                    model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                    messages=st.session_state.current_chat
                )

                # Handle response from Together API (no raw response output)
                if 'choices' in response and len(response['choices']) > 0:
                    answer = response['choices'][0]['message']['content'].strip()
                else:
                    st.error(f"Unexpected response format: {response}")
                    answer = "Sorry, I couldn't get a response from the model."

            st.session_state.current_chat.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Error occurred while fetching response: {str(e)}")
        finally:
            end_time = time.time()
            st.info(f"API call duration: {end_time - start_time:.2f} seconds")

        # Refresh the conversation to display the entire chat history in real time
        conversation_placeholder.empty()  # Clear the existing conversation
        with conversation_placeholder.container():
            for msg in st.session_state.current_chat:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:  # Ensure msg is a dictionary
                    if msg["role"] == "user":
                        st.markdown(f"**🧑 User:** {msg['content']}")
                    elif msg["role"] == "assistant":
                        st.markdown(f"**🤖 Botify:** {msg['content']}")
                else:
                    st.error("Error: A message is missing in the current chat history.")

    # Save chat history to file
    save_chat_history(st.session_state.chat_history)

# Display full chat history dynamically in a collapsible container
with st.expander("Chat History"):
    for i, conversation in enumerate(st.session_state.chat_history):
        st.write(f"**Conversation {i+1}:**")
        for msg in conversation:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:  # Ensure msg is a valid dictionary
                role = "User" if msg["role"] == "user" else "Botify"
                st.write(f"**{role}:** {msg['content']}")
            else:
                st.error("Error: A message is missing or malformed in the chat history.")
