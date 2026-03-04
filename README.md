# 🤖 Acadbot — AI Research Thesis Writer

An AI-powered academic research assistant that generates complete IEEE-format LaTeX thesis papers with auto-generated neural network diagrams and literature charts.

## ✨ Features

- **📝 Thesis Writer** — Enter a topic, get a full LaTeX research paper with word-by-word streaming
  - Qwen fine-tuned model for domain-specific chest X-ray research content
  - Gemini API for structuring into publication-ready IEEE LaTeX format
  - Auto-generated NN architecture diagrams (DenseNet121, ResNet50, VGG16, MobileNetV2, InceptionV3)
  - Auto-generated literature trend charts from your dataset
  - Humanized writing style to avoid AI detection
  - Download as `.zip` (LaTeX + figures)

- **📊 Visualizer** — Generate charts, graphs, and 3D neural network architecture diagrams

- **📄 Document Reader** — Upload PDFs/text files and ask questions about them

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| NLP Model | Qwen (fine-tuned, 4-bit NF4 quantized) |
| API | Google Gemini 2.5 Flash |
| NN Diagrams | VisualKeras + TensorFlow |
| Charts | Plotly |
| Frontend | Vanilla HTML/CSS/JS |

## 🚀 Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/acadbot.git
cd acadbot
```

### 2. Install dependencies
```bash
pip install flask torch transformers peft accelerate bitsandbytes
pip install google-genai python-dotenv
pip install plotly kaleido pandas visualkeras tensorflow
pip install PyPDF2
```

### 3. Configure environment
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Add model weights
Download/train the Qwen model and place it in `./qwen-xray-researcher/`

### 5. Add dataset
Place your `literature_dataset.json` in the project root.

### 6. Run
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 📁 Project Structure

```
acadbot/
├── app.py                  # Flask backend with all API routes
├── static/
│   ├── style.css           # UI styling
│   └── script.js           # Frontend logic
├── templates/
│   └── index.html          # Main page
├── colab_text_train.py     # Training script for Colab
├── colab_train.py          # Alternative training script
├── prepare_qa_pairs.py     # Dataset preparation
├── build_vector_db.py      # Vector DB builder
├── fetch_literature.py     # Literature fetcher
├── download_dataset.py     # Dataset downloader
├── .env                    # API keys (not tracked)
└── .gitignore
```

## 📄 License

MIT
