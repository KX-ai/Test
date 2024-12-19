import os
import openai
import requests
import streamlit as st
import json
import time
import fitz  # PyMuPDF (corrected import)
import tempfile

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Maximum context length for DeepSeek
MAX_CONTEXT_LENGTH = 4096

# Define a safe buffer for max completion tokens
MAX_COMPLETION_TOKENS = 300
SAFE_CONTEXT_LENGTH = MAX_CONTEXT_LENGTH - MAX_COMPLETION_TOKENS

# Use the Sambanova API for Qwen 2.5-72B-Instruct
class SambanovaClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        openai.api_key = self.api_key
        openai.api_base = self.base_url

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
            raise Exception(f"Error while calling Sambanova API: {str(e)}")

# Use the DeepSeek API for DeepSeek LLM Chat (67B)
class DeepSeekClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.together.xyz/v1/chat/completions"
        self.rate_limit_retry_delay = 1  # Retry delay in seconds for rate-limiting errors

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

        # Retry logic for rate-limiting errors
        for attempt in range(5):  # Retry up to 5 times
            try:
                response = requests.post(self.url, json=payload, headers=headers)
                response_data = response.json()

                if response.status_code == 429:  # Rate limit error
                    time.sleep(self.rate_limit_retry_delay)
                    continue

                if response.status_code != 200 or "choices" not in response_data:
                    raise Exception(f"Error: {response_data.get('error', 'Unknown error')}")
                return response_data

            except Exception as e:
                if attempt == 4:  # Final attempt failed
                    raise Exception(f"Error while calling DeepSeek API: {str(e)}")
                time.sleep(self.rate_limit_retry_delay)

# Function to extract text from PDF using PyMuPDF (corrected import)
@st.cache_data
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())  
        temp_file_path = temp_file.name

    doc = fitz.open(temp_file_path)
    text = ""
    for page_num in range(len(doc)):  
        page = doc.load_page(page_num)  
        text += page.get_text()  
    os.remove(temp_file_path)  
    return text

# Function to truncate text for token limit
def truncate_text(text, max_length):
    return text[:max_length]

# Function to load chat history from a JSON file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []

# Function to save chat history to a JSON file
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Function to manage token limit for messages
def truncate_chat_history(chat_history, pdf_context, max_length):
    total_length = len(pdf_context)
    truncated_history = []

    # Add PDF context to the chat history
    if pdf_context:
        truncated_history.append({"role": "system", "content": pdf_context})
        total_length += len(pdf_context)

    # Add the previous conversation messages to the history
    for message in reversed(chat_history):
        message_text = message["content"]
        total_length += len(message_text)
        if total_length > max_length:
            break
        truncated_history.insert(0, message)
    
    return truncated_history

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Display success message once the PDF is uploaded and processed
if pdf_file:
    text_content = extract_text_from_pdf(pdf_file)
    st.success("PDF text extracted successfully.")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]

# Display the hello message at the start
if not st.session_state.chat_history:
    st.session_state.chat_history.append(st.session_state.current_chat)

# Button to start a new chat
if st.button("Start New Chat"):
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! Starting a new conversation. How can I assist you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    save_chat_history(st.session_state.chat_history)
    st.success("New chat started!")

# Button to delete a specific conversation
def delete_conversation(idx):
    if len(st.session_state.chat_history) > idx:
        st.session_state.chat_history.pop(idx)
        save_chat_history(st.session_state.chat_history)
        st.experimental_rerun()

# Display chat history with delete button for each conversation
st.write("### Chat History")
for idx, conversation in enumerate(st.session_state.chat_history):
    st.write(f"**Conversation {idx + 1}:**")
    for msg in conversation:
        if msg["role"] == "user":
            st.markdown(f"*\U0001F9D1 User:* {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"*\U0001F916 Botify:* {msg['content']}")
    delete_button = st.button(f"Delete Conversation {idx + 1}", key=f"delete_{idx}")
    if delete_button:
        delete_conversation(idx)

# API keys
sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
deepseek_api_key = st.secrets["general"]["DEEPSEEK_API_KEY"]

# Model selection
model_choice = st.selectbox("Select the LLM model:", ["Sambanova (Qwen 2.5-72B-Instruct)", "DeepSeek LLM Chat (67B)"])

# Flag to check if user has entered input before generating responses
user_input = st.text_input("Your message:", key="user_input", placeholder="Type your message here and press Enter")

# Only proceed if the user input is not empty
if user_input:
    st.session_state.current_chat.append({"role": "user", "content": user_input})

    # Prepare the prompt based on uploaded PDF content
    if pdf_file:
        truncated_text = truncate_text(text_content, SAFE_CONTEXT_LENGTH // 2)  # Limit PDF context size
    else:
        truncated_text = ""

    # Truncate chat history to fit token limits, including the PDF content
    truncated_history = truncate_chat_history(st.session_state.chat_history, truncated_text, SAFE_CONTEXT_LENGTH)

    try:
        if model_choice == "Sambanova (Qwen 2.5-72B-Instruct)":
            sambanova_client = SambanovaClient(
                api_key=sambanova_api_key,
                base_url="https://api.sambanova.ai/v1"
            )
            response = sambanova_client.chat(
                model="Qwen2.5-72B-Instruct",
                messages=truncated_history + [{"role": "user", "content": user_input}],
                temperature=0.1,
                top_p=0.1,
                max_tokens=MAX_COMPLETION_TOKENS
            )
            answer = response['choices'][0]['message']['content'].strip()

        elif model_choice == "DeepSeek LLM Chat (67B)":
            deepseek_client = DeepSeekClient(api_key=deepseek_api_key)
            response = deepseek_client.chat(
                model="deepseek-ai/deepseek-llm-67b-chat",
                messages=truncated_history + [{"role": "user", "content": user_input}]
            )
            answer = response.get('choices', [{}])[0].get('message', {}).get('content', "No response received.")

        # Append the assistant's response
        st.session_state.current_chat.append({"role": "assistant", "content": answer})
        save_chat_history(st.session_state.chat_history)
        st.experimental_rerun()

    except Exception as e:
        st.error(f"Error while fetching response: {e}")
        if "rate limit" in str(e).lower():
            st.warning("Rate limit exceeded. Please wait a moment and try again.")

# Save chat history
save_chat_history(st.session_state.chat_history)
