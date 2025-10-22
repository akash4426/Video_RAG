🎥 Video RAG: Semantic Video Search & Summarization

🚀 CLIP + FAISS + Gemini = Smarter Video Understanding

A Streamlit-based web app that lets you semantically search within videos and get AI-powered summaries — turning your videos into searchable, explainable knowledge.

<p align="center"> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/Model-CLIP-blue?style=for-the-badge&logo=openai&logoColor=white"/> <img src="https://img.shields.io/badge/Search-FAISS-green?style=for-the-badge&logo=facebook&logoColor=white"/> <img src="https://img.shields.io/badge/API-Gemini AI-yellow?style=for-the-badge&logo=google&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/> </p>
📚 Table of Contents

✨ Features :

⚙️ How It Works

🚀 Getting Started

🔑 Configuration

🧠 Key Libraries Used

🧩 Example Use Case

🤝 Contributing

🧾 License

💡 Author

✨ Features

✅ Video Upload – Supports .mp4, .mov, and .avi formats
⚙️ Custom Frame Sampling – Adjust FPS to balance speed and detail
🔍 Semantic Search – Query in natural language (e.g., “a person riding a horse”)
🖼️ Frame Retrieval – Instantly shows top 3 matching frames with timestamps
🧠 Gemini Summaries – Generates short, intelligent summaries based on your query and frames

⚙️ How It Works

The app follows a powerful yet simple Video RAG pipeline:

🎞️ Extract Frames: Sample video frames using OpenCV

🧩 Generate Embeddings: Encode frames using CLIP to capture visual meaning

🧮 Index with FAISS: Store embeddings for ultra-fast similarity search

🔎 Retrieve Matches: Compare your text query against all frame embeddings

💬 Summarize (Optional): Send top frames and query to Gemini API for a natural language summary

🚀 Getting Started
🧩 Prerequisites

Python 3.8+

Google Gemini API Key (for AI summaries)

1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/Video-RAG.git
cd Video-RAG

2️⃣ Create a Virtual Environment
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py


🌐 Your app will open automatically in your default browser.

🔑 Configuration

To enable the AI Summary feature:

Enter your Google Gemini API Key in the input field at the top of the app.

Get your key from Google AI Studio
.

🧠 Key Libraries Used
Library	Purpose
Streamlit	Interactive web interface
OpenCV (cv2)	Frame extraction and processing
PyTorch	Backend for CLIP embeddings
Transformers (Hugging Face)	Pre-trained CLIP model
FAISS (Facebook AI)	Vector similarity search
Pillow (PIL)	Image manipulation
Google Generative AI	Gemini API integration
🧩 Example Use Case

Upload a short video (e.g., “city street.mp4”)

Type a query like “a red car crossing the road”

Get the top 3 matching frames + timestamps

Let Gemini generate a summary like:

“A red sedan drives through a crowded urban street during daylight.”

🎬 Demo Preview (Add Your GIF Here)
<p align="center"> <img src="https://github.com/YOUR_USERNAME/Video-RAG/assets/demo.gif" width="700"/> </p>

(Tip: Record a quick 10s screen capture using ScreenPal or Kap, then upload it to your repo as demo.gif)

🤝 Contributing

Contributions, pull requests, and ideas are welcome! 💡
If you find a bug or have a feature request, please open an issue.

💡 Author

👤 Akash Karri
📧 akashkarri2006@gmail.com

