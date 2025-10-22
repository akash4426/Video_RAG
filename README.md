🎥 Video RAG: Semantic Video Search & Summarization

This project is a Streamlit web application that allows you to perform semantic search over video content. It uses CLIP to understand the content of video frames and FAISS for efficient similarity search, creating a powerful Video-based Retrieval-Augmented Generation (RAG) system.

Additionally, it integrates with the Gemini API to provide AI-powered summaries of the retrieved video segments.

✨ Features

Video Upload: Upload your own mp4, mov, or avi video files.

Custom Frame Sampling: Adjust the sampling rate (Frames Per Second) to balance speed and accuracy.

Semantic Text Search: Use natural language queries (e.g., "a person walking a dog") to find relevant moments in the video.

Frame Retrieval: Instantly displays the top 3 matching frames from the video, along with their timestamps and similarity scores.

AI-Powered Summaries: Uses Google's Gemini API to generate a concise summary based on your query and the content of the retrieved frames.

⚙️ How It Works

The application follows a simple but powerful pipeline:

Extract Frames: The uploaded video is sampled at the specified FPS, and individual frames are extracted.

Generate Embeddings: Each frame is processed by the CLIP model, which generates a vector embedding that captures the semantic meaning of the image.

Build Index: The frame embeddings are indexed into a FAISS vector database. This allows for ultra-fast similarity search.

Retrieve: When you enter a text query, CLIP generates an embedding for the text. FAISS then compares this text embedding against all frame embeddings to find the closest matches.

Summarize (Optional): The top matching frames are sent to the Gemini multimodal API, along with the original query, to generate a descriptive summary.

🚀 Getting Started

Prerequisites

Python 3.8+

A Google Gemini API Key (for the summarization feature)

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME


2. Create a Virtual Environment (Recommended)

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate


3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt


4. Run the Application

Once the dependencies are installed, run the Streamlit app:

streamlit run video_rag_gemini.py


Your app will open automatically in your web browser.

🔑 Configuration

To use the AI summary feature, you must provide your Google Gemini API Key in the text input field at the top of the app.

You can get an API key from the Google AI Studio.

🛠️ Key Libraries Used

Streamlit: For building the interactive web UI.

OpenCV (cv2): For video processing and frame extraction.

PyTorch: The deep learning framework (runs CLIP).

Transformers (Hugging Face): For loading the pre-trained CLIP model.

FAISS (Facebook AI): For efficient vector search.

Pillow (PIL): For image manipulation.

Google Generative AI: For interfacing with the Gemini API.

Feel free to contribute, open issues, or suggest improvements!
