import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import io
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# Function to load the GPT-Neo model using the pipeline
@st.cache_resource
def load_model():
    try:
        # Ensure GPU utilization if available
        return pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=0)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
generator = load_model()

# Function to upload PDF file
def upload_file():
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        st.write(f"File uploaded: {uploaded_file.name}")
    return uploaded_file

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=io.BytesIO(pdf_file.read()), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to chunk text
def chunk_text(text, chunk_size=2048):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to summarize text
def summarize_text(long_text):
    if not generator:
        return "Model could not be loaded. Please try again later."
    prompt = f"""Summarize the following text:

{long_text[:2048]}"""

    try:
        outputs = generator(prompt, do_sample=True, max_new_tokens=100, pad_token_id=50256)
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return ""

# Function to interact with GPT-Neo
def query_gpt_neox(text_prompt):
    if not generator:
        return "Model could not be loaded. Please try again later."
    try:
        outputs = generator(text_prompt, do_sample=True, max_new_tokens=100, pad_token_id=50256)
        return outputs[0]["generated_text"].strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# Function to process chunks in parallel
def process_chunks_in_parallel(chunks, user_input):
    def process_chunk(chunk):
        prompt = (
            f"Based on the following document text, answer the user's question:\n\n"
            f"Document Text: {chunk}\n\n"
            f"User Question: {user_input}\nAnswer:"
        )
        return query_gpt_neox(prompt)

    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(process_chunk, chunks))
    return responses

# Function to summarize chunks
def summarize_chunks(chunks):
    summarized_chunks = []
    for chunk in chunks:
        summarized_chunk = summarize_text(chunk)
        summarized_chunks.append(summarized_chunk)
    return summarized_chunks

# Function to combine responses
def combine_responses(responses):
    combined_text = " ".join(responses)
    return summarize_text(combined_text)  # Summarize the combined responses

# Function for adaptive chunk size
def adaptive_chunk_size(text, max_length=2048, complexity_threshold=1000):
    if len(text.split()) > complexity_threshold:
        return chunk_text(text, chunk_size=1024)  # Use smaller chunks for complex text
    return chunk_text(text, chunk_size=max_length)

# Function to handle user input and generate response based on document text
def chat_with_document(document_text, user_input):
    # Chunk the document text adaptively
    chunks = adaptive_chunk_size(document_text)

    # Summarize each chunk
    summarized_chunks = summarize_chunks(chunks)

    # Process chunks in parallel
    responses = process_chunks_in_parallel(summarized_chunks, user_input)

    # Combine responses and summarize the final answer
    final_response = combine_responses(responses)
    return final_response

# Main Streamlit app layout
def main():
    st.title("Interactive Chat with Document Content")

    # Check if the model loaded correctly
    if not generator:
        st.error("The GPT-Neo model could not be loaded. Please restart the app.")
        return

    # Upload PDF file
    uploaded_file = upload_file()

    if uploaded_file:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            document_text = extract_text_from_pdf(uploaded_file)
        if document_text:
            st.success("Document uploaded and text extracted successfully!")

            # Chat feature - ask a question about the document
            user_input = st.text_input("Ask a question about the document:")

            if user_input:
                # Generate response using GPT-Neo
                with st.spinner("Generating response..."):
                    response = chat_with_document(document_text, user_input)
                st.success("Response generated!")
                st.write(f"Answer: {response}")
            else:
                st.write("Please enter a question to ask.")
        else:
            st.error("Failed to extract text from the document.")
    else:
        st.write("Please upload a PDF to start the conversation.")

if __name__ == "__main__":
    main()
