import os
import openai
import requests
import json
import streamlit as st
import PyPDF2

# File path for saving chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Maximum context length
MAX_CONTEXT_LENGTH = 8192

# Sambanova Client (Qwen Model)
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

# Groq API Client (Gemma Model)
class GroqClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.groq.com/v1/chat/completions"  # Replace with correct URL if necessary

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
            response_data = response.json()
            if response.status_code != 200 or "choices" not in response_data:
                raise Exception(f"Error: {response_data.get('error', 'Unknown error')}")
            return response_data
        except Exception as e:
            raise Exception(f"Error while calling Groq API: {str(e)}")

# Function to extract text from PDF using PyPDF2
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    else:
        return []

# Save chat history
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Truncate long texts
def truncate_text(text, max_length=MAX_CONTEXT_LENGTH):
    return text[:max_length]

# Streamlit UI setup
st.set_page_config(page_title="Chatbot with PDF (Botify)", layout="centered")
st.title("Botify")

# Upload a PDF file
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! I am Botify, your assistant. How can I assist you today?"}]

# Button to start a new chat
if st.button("Start New Chat"):
    st.session_state.current_chat = [{"role": "assistant", "content": "Hello! Starting a new conversation. How can I assist you today?"}]
    st.session_state.chat_history.append(st.session_state.current_chat)
    st.success("New chat started!")

# Display chat dynamically
st.write("### Chat Conversation")
for msg in st.session_state.current_chat:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        if msg["role"] == "user":
            st.markdown(f"**\U0001F9D1 User:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**\U0001F916 Botify:** {msg['content']}")

# API keys
sambanova_api_key = st.secrets["general"]["SAMBANOVA_API_KEY"]
groq_api_key = st.secrets["general"]["GROQ_API_KEY"]

# Model selection
model_choice = st.selectbox("Select the LLM model:", ["Sambanova (Qwen 2.5-72B-Instruct)", "Groq (Gemma-2-9B-IT)"])

# Input message
user_input = st.text_area("Your message:", key="user_input", placeholder="Type your message here and press Enter")

if user_input:
    if user_input != st.session_state.current_chat[-1]["content"]:
        st.session_state.current_chat.append({"role": "user", "content": user_input})

    if pdf_file:
        text_content = extract_text_from_pdf(pdf_file)
        truncated_text = truncate_text(text_content)
        prompt_text = f"Document content:\n{truncated_text}\n\nUser question: {user_input}\nAnswer:"
    else:
        prompt_text = f"User question: {user_input}\nAnswer:"

    st.session_state.current_chat.append({"role": "system", "content": prompt_text})

    try:
        if model_choice == "Sambanova (Qwen 2.5-72B-Instruct)":
            response = SambanovaClient(
                api_key=sambanova_api_key,
                base_url="https://api.sambanova.ai/v1"
            ).chat(
                model="Qwen2.5-72B-Instruct",
                messages=st.session_state.current_chat,
                temperature=0.1,
                top_p=0.1,
                max_tokens=300
            )
            answer = response['choices'][0]['message']['content'].strip()
        elif model_choice == "Groq (Gemma-2-9B-IT)":
            response = GroqClient(api_key=groq_api_key).chat(
                model="gemma-2-9b-it",  # Specify the correct model name if different
                messages=st.session_state.current_chat
            )
            answer = response.get('choices', [{}])[0].get('message', {}).get('content', "No response received.")
        
        st.session_state.current_chat.append({"role": "assistant", "content": answer})
        st.experimental_rerun()  # Rerun to update chat dynamically

    except Exception as e:
        st.error(f"Error while fetching response: {e}")

# Save chat history
save_chat_history(st.session_state.chat_history)

# Display chat history with deletion option
with st.expander("Chat History"):
    for i, conversation in enumerate(st.session_state.chat_history):
        with st.container():
            st.write(f"**Conversation {i + 1}:**")
            for msg in conversation:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = "User" if msg["role"] == "user" else "Botify"
                    st.write(f"**{role}:** {msg['content']}")
            if st.button(f"Delete Conversation {i + 1}", key=f"delete_{i}"):
                del st.session_state.chat_history[i]
                save_chat_history(st.session_state.chat_history)
                st.experimental_rerun()
