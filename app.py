import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import faiss
import os

# Function to convert speech to text
def convert_speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_path)

    # Normalize the audio to reduce noise
    audio = normalize(audio)

    # Export the entire audio to a temporary WAV file
    audio.export("temp.wav", format="wav")

    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Load pre-trained models for embedding and response generation
embedder_name = "sentence-transformers/all-MiniLM-L6-v2"  # BERT-based embedding model
generator_name = "google/flan-t5-base"  # Use a larger model for better summarization

# Initialize tokenizer and models
embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_name)
embedder_model = AutoModel.from_pretrained(embedder_name)

# Setup text generation pipeline
generator = pipeline("summarization", model=generator_name, tokenizer=generator_name)

def embed_text(text):
    inputs = embedder_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        embeddings = embedder_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Create a FAISS index for similarity search
index = faiss.IndexFlatL2(384)  # Assuming 384-dimensional embeddings

def add_to_index(texts):
    for text in texts:
        embedding = embed_text(text)
        index.add(embedding)

def retrieve_similar_context(query):
    query_embedding = embed_text(query)
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 similar contexts
    return I[0]  # Extract the first row to get the indices

# Streamlit app interface
st.title("Audio File Contextual Chatbot")
st.write("Upload an audio file to get a context-based response.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    if st.button("Submit"):
        with st.spinner("Transcribing audio..."):
            # Save the uploaded file to a temporary location
            temp_file_path = "uploaded_audio.wav"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convert speech to text
            transcribed_text = convert_speech_to_text(temp_file_path)
            st.write("Transcribed Text:", transcribed_text)

            # Embed the transcribed text and store it in the FAISS index
            add_to_index([transcribed_text])

        with st.spinner("Generating summary..."):
            # Retrieve similar contexts
            similar_contexts = retrieve_similar_context(transcribed_text)
            context_texts = [transcribed_text]  # For simplicity, we are using the transcribed text directly

            # Generate a summary using the language model
            prompt = " ".join(context_texts)
            response = generator(prompt, max_length=150, min_length=30, num_return_sequences=1)
            response_text = response[0]['summary_text']

            st.write("Summary:", response_text)

            # Clean up temporary file
            os.remove(temp_file_path)
