
# Audio File Contextual Chatbot

## Overview

This project implements a contextual chatbot using Streamlit, which transcribes audio files and generates context-based summaries. It uses the Google Web Speech API for transcription, FAISS for similarity search, and a Flan-T5 model for summarization.

## Features

- **Audio Transcription**: Converts audio files to text using the Google Web Speech API.
- **Text Embedding**: Uses a BERT-based model to embed the transcribed text for similarity search.
- **Context Retrieval**: Employs FAISS to retrieve similar text contexts based on embeddings.
- **Summarization**: Generates summaries of the transcribed text using a Flan-T5 model.
- **Interactive Interface**: Provides an easy-to-use web interface for uploading audio files and viewing summaries.

## Setup and Installation

### Prerequisites

- Python 3.9 or later
- Conda (for managing environments)

### Installation

1. **Clone the Repository**

   Clone the repository to your local machine:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Conda Environment**

   Create and activate a new Conda environment for the project:

   ```bash
   conda create --name streamlit_env python=3.9
   conda activate streamlit_env
   ```

3. **Install Required Libraries**

   Install the necessary Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg for Audio Processing**

   FFmpeg is required for processing audio files. You can install it using Homebrew on macOS:

   ```bash
   brew install ffmpeg
   ```

### Running the Application

1. **Start the Streamlit App**

   Navigate to the project directory and run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. **Open the Web Interface**

   Open the provided URL in your web browser to access the application.

3. **Upload and Process Audio Files**

   - Use the file uploader to upload an audio file (WAV, MP3, OGG, or FLAC).
   - Click the "Submit" button to transcribe the audio and generate a summary.
   - View the transcribed text and generated summary in the web interface.

## Dependencies

The project requires the following Python libraries:

- `streamlit`: For creating the web interface.
- `faiss-cpu`: For similarity search and clustering of dense vectors.
- `speechrecognition`: For converting speech to text.
- `pydub`: For handling audio file processing.
- `transformers`: For using pre-trained models for text embedding and summarization.
- `torch`: For model inference and computation.

## Acknowledgments

- **Google Web Speech API**: Used for audio-to-text transcription.
- **Hugging Face Transformers**: Provides pre-trained models for text processing and summarization.
- **Facebook AI Similarity Search (FAISS)**: Enables efficient similarity search of dense vectors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
