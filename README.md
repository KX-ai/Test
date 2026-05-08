# 🤖 Botify

A multimodal AI chatbot built with Streamlit that lets you interact with PDFs, images, and audio files using large language models via the Groq API.

---

## Features

- **📄 PDF Chat** — Upload a PDF, get an AI-generated summary, translate it into 28+ languages, and ask follow-up questions about the content.
- **🖼️ Image Understanding** — Upload an image and extract a natural language description using the BLIP image captioning model (Salesforce/blip-image-captioning-large).
- **🎙️ Audio Transcription** — Upload an audio file and transcribe it using Groq's Whisper large-v3-turbo model, then chat with the transcript.
- **💬 Conversational Q&A** — Ask questions about any uploaded content and get context-aware answers powered by Groq-hosted LLMs.
- **📊 ROUGE Scoring** — Automatically evaluates summarization and Q&A quality using ROUGE-1, ROUGE-2, and ROUGE-L metrics.
- **🕒 Response Timing** — Displays response and summarization time for each query.
- **🗂️ Conversation History** — View, switch between, and manage past chat sessions via the sidebar.
- **🔊 Text-to-Speech** — Summaries are converted to audio using gTTS and played back in the app.

---

## Supported Models

| Name | Model ID |
|---|---|
| Mixtral 8x7b | `mixtral-8x7b-32768` |
| Llama 3.1 8b Instant | `llama-3.1-8b-instant` |
| Gemma2 9b | `gemma2-9b-it` |

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — UI framework
- [Groq API](https://groq.com/) — LLM inference (chat + Whisper transcription)
- [Hugging Face Transformers](https://huggingface.co/) — BLIP image captioning
- [PyPDF2](https://pypi.org/project/PyPDF2/) — PDF text extraction
- [gTTS](https://pypi.org/project/gTTS/) — Text-to-speech
- [rouge-score](https://pypi.org/project/rouge-score/) — Evaluation metrics

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/)
- A [Hugging Face token](https://huggingface.co/settings/tokens) (for BLIP model access)

### Installation

```bash
git clone https://github.com/KX-ai/Test.git
cd Test
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file in the project root with the following:

```toml
[groq_api]
api_key = "your_groq_api_key_here"

[whisper]
WHISPER_API_KEY = "your_groq_api_key_here"
```

> **Note:** Both keys point to the same Groq API key. The Hugging Face token is currently hardcoded in `app.py` — it is recommended to move it to `secrets.toml` as well before deploying.

### Running the App

```bash
streamlit run app.py
```

---

## Usage

1. Select an input method from the dropdown: **Upload PDF**, **Upload Audio**, or **Upload Image**.
2. Upload your file using the file uploader.
3. For PDFs:
   - Choose a language model and output language.
   - Click **Summarize Text** to generate a summary with audio playback and translation.
4. Use the chat input at the bottom to ask questions about any uploaded content.
5. View past conversations in the sidebar, switch between sessions, or start a new chat.

---

## Supported Audio Formats

`flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `opus`, `wav`, `webm`

## Supported Translation Languages

28 languages including English, Malay, Chinese, Spanish, French, Arabic, Japanese, Korean, Hindi, and more.

---

## Project Structure

```
Test/
├── app.py              # Main Streamlit application
└── requirements.txt    # Python dependencies
```

---

## Dependencies

```
requests==2.32.3
streamlit==1.41.1
PyPDF2
pytesseract
Pillow
pytz
python-dotenv
openai
pydub
transformers
gTTS
numpy
google-generativeai
torch
datasets==2.4.0
rouge-score
```

---

## License

This project is open source. Feel free to fork and build on it.
