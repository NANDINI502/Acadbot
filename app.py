import os
import torch
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_file, Response
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import pandas as pd
import plotly.express as px
import plotly.io as pio
from google import genai
from google.genai import types
from dotenv import load_dotenv
import visualkeras
import tensorflow as tf
from collections import defaultdict
import chromadb
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps
import jwt

load_dotenv()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# --- Firebase Admin Initialization ---
try:
    if os.path.exists("serviceAccountKey.json"):
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    else:
        firebase_admin.initialize_app(options={'projectId': 'acadbot-dev-01'})
except ValueError:
    pass  # Already initialized
except Exception as e:
    print(f"Warning: Firebase Admin SDK failed to initialize. Auth won't work: {e}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split('Bearer ')[1]
        elif 'token' in request.args:
            token = request.args.get('token')
            
        if not token:
            return jsonify({"error": "Unauthorized. Please authenticate."}), 401
            
        try:
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
        except Exception as e:
            if "default credentials were not found" in str(e):
                print("Warning: Firebase ADC missing. Bypassing signature verification for local dev!")
                try:
                    request.user = jwt.decode(token, options={"verify_signature": False})
                except Exception as decode_e:
                    return jsonify({"error": f"Invalid token format: {str(decode_e)}"}), 401
            else:
                print(f"Token verification failed: {e}")
                return jsonify({"error": f"Invalid token: {str(e)}"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# --- Configuration ---
TEXT_MODEL_DIR = "./qwen-xray-researcher"
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Globals ---
text_tokenizer = None
text_model = None
_diagram_model_cache = {}  # Cache built Keras models for diagram generation
_cached_literature_dataset = None  # Cache 10MB JSON dataset in RAM

# --- ChromaDB + Embedding model for fast reference search ---
print("Loading embedding model for reference search...")
_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
_chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    _chroma_collection = _chroma_client.get_collection("medical_literature")
    print(f"ChromaDB loaded: {_chroma_collection.count()} chunks available")
except Exception:
    _chroma_collection = None
    print("ChromaDB collection 'medical_literature' not found — falling back to JSON search")

print("Pre-loading literature dataset into RAM...")
dataset_path = "literature_dataset.json"
if os.path.exists(dataset_path):
    try:
        if dataset_path.endswith(".csv"):
            _cached_literature_dataset = pd.read_csv(dataset_path).to_dict(orient="records")
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                _cached_literature_dataset = json.load(f)
        print(f"Loaded {len(_cached_literature_dataset)} records into RAM cache.")
    except Exception as e:
        print(f"Failed to cache dataset: {e}")

# ============================================================
# BACKEND LOGIC (kept from original)
# ============================================================

def load_text_model():
    global text_tokenizer, text_model
    if text_model is not None: return True, "Text model already loaded."
    if not os.path.exists(TEXT_MODEL_DIR):
        return False, f"Model directory '{TEXT_MODEL_DIR}' not found."
    try:
        from transformers import BitsAndBytesConfig
        print(f"Loading Fine-Tuned NLP Model from: {TEXT_MODEL_DIR} in 4-bit (NF4)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
        
        # Fine-Tuned PEFT Qwen Model
        text_model = AutoPeftModelForCausalLM.from_pretrained(
            TEXT_MODEL_DIR, 
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa" # PyTorch 2.0+ Scaled Dot Product Attention
        )
        text_model.eval()
        return True, "Text model loaded successfully!"
    except Exception as e:
        return False, f"Error loading text model: {str(e)}"

def generate_draft(prompt, max_tokens=800, temperature=0.7):
    success, msg = load_text_model()
    if not success:
        return f"🚨 {msg}"
    messages = [
        {"role": "system", "content": "You are a distinguished AI medical researcher writing a peer-reviewed academic paper."},
        {"role": "user", "content": prompt}
    ]
    input_text = text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = text_tokenizer(input_text, return_tensors="pt").to(text_model.device)
    with torch.no_grad():
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(text_tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Run the heavy PyTorch generation in a separate thread so it doesn't block the Flask SSE stream
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                text_model.generate,
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, repetition_penalty=1.1, streamer=streamer
            )
            generated_text = ""
            for chunk in streamer:
                generated_text += chunk
                print(chunk, end="", flush=True)  # Print to terminal so user sees progress
            print("") # newline
            future.result()
            
    return generated_text.split("Keywords:")[0].strip()

def analyze_data(query, dataset_path="literature_dataset.json"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None, "🚨 No Gemini API Key found in .env"
    
    global _cached_literature_dataset
    if _cached_literature_dataset is not None:
        full_data = _cached_literature_dataset
    else:
        if not os.path.exists(dataset_path): return None, f"🚨 '{dataset_path}' not found."
        try:
            if dataset_path.endswith(".csv"):
                full_data = pd.read_csv(dataset_path).to_dict(orient="records")
            else:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    full_data = json.load(f)
        except Exception as e:
            return None, f"🚨 Error reading dataset: {str(e)}"

    try:
        sample_data = full_data[:50]
        schema_context = f"List of dicts, {len(full_data)} records."
        prompt = f"""You are an expert Data Scientist analyzing a dataset.
        Dataset Context: {schema_context}
        Sample: {json.dumps(sample_data, indent=2)}
        User asks: "{query}"
        Write Python that:
        1. Loads '{dataset_path}':
        `import json, pandas as pd`
        `if '{dataset_path}'.endswith('.csv'): df = pd.read_csv('{dataset_path}')`
        `else:`
        `  with open('{dataset_path}', 'r', encoding='utf-8') as f:`
        `      df = pd.DataFrame(json.load(f))`
        2. Creates EITHER `fig` (Plotly) OR `markdown_output` (string).
        Chart styling: template="plotly_white", title_x=0.5, margin=dict(t=100,l=40,r=40,b=40).
        ONLY return raw Python code, no markdown backticks."""
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        code = response.text
        for prefix in ["```python", "```"]:
            if code.startswith(prefix): code = code[len(prefix):]
        if code.endswith("```"): code = code[:-3]
        code = code.strip()
        ns = globals().copy()
        exec(code, ns)
        if 'fig' in ns: return ns['fig'], None
        elif 'markdown_output' in ns: return None, ns['markdown_output']
        else: return None, "🚨 No output generated."
    except Exception as e:
        return None, f"🚨 Error: {str(e)}"

def generate_model_diagram(model_name):
    try:
        model_map = {
            "DenseNet121": tf.keras.applications.DenseNet121,
            "ResNet50": tf.keras.applications.ResNet50,
            "VGG16": tf.keras.applications.VGG16,
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "InceptionV3": tf.keras.applications.InceptionV3
        }
        if model_name not in model_map: return None

        # Use cached model if available, otherwise build and cache it
        if model_name in _diagram_model_cache:
            print(f"[Diagram] Using cached {model_name} model")
            model = _diagram_model_cache[model_name]
        else:
            print(f"[Diagram] Building {model_name} model (will be cached)...")
            model = model_map[model_name](weights=None, include_top=True)
            # Monkey-patch: TF 2.20 removed output_shape from InputLayer
            for layer in model.layers:
                if not hasattr(layer, 'output_shape'):
                    try:
                        layer.output_shape = layer.output.shape
                    except Exception:
                        layer.output_shape = (None, 224, 224, 3)
            _diagram_model_cache[model_name] = model

        color_map = defaultdict(dict)
        color_map[tf.keras.layers.Conv2D]['fill'] = '#00f5d4'
        color_map[tf.keras.layers.MaxPooling2D]['fill'] = '#8338ec'
        color_map[tf.keras.layers.Dense]['fill'] = '#ff006e'
        color_map[tf.keras.layers.Flatten]['fill'] = '#ffbe0b'
        color_map[tf.keras.layers.GlobalAveragePooling2D]['fill'] = '#3a86ff'
        color_map[tf.keras.layers.Dropout]['fill'] = '#e0afa0'
        color_map[tf.keras.layers.BatchNormalization]['fill'] = '#fb5607'
        color_map[tf.keras.layers.Activation]['fill'] = '#ff006e'
        color_map[tf.keras.layers.ZeroPadding2D]['fill'] = '#606080'
        color_map[tf.keras.layers.AveragePooling2D]['fill'] = '#3a86ff'
        color_map[tf.keras.layers.Concatenate]['fill'] = '#ffbe0b'
        
        # For large models, hide filler layers so the diagram is readable
        num_layers = len(model.layers)
        ignore_types = []
        if num_layers > 100:
            # Hide BatchNorm, Activation, ZeroPadding, Concatenate — they clutter
            ignore_types = [
                tf.keras.layers.BatchNormalization,
                tf.keras.layers.Activation,
                tf.keras.layers.ZeroPadding2D,
                tf.keras.layers.Concatenate,
                tf.keras.layers.Add,
            ]
        
        if num_layers > 200:
            spacing = 10
            scale_xy = 1
            scale_z = 1
            max_z = 50
        elif num_layers > 50:
            spacing = 15
            scale_xy = 1.5
            scale_z = 1
            max_z = 60
        else:
            spacing = 30
            scale_xy = 2
            scale_z = 1
            max_z = 80
        
        img = visualkeras.layered_view(
            model, legend=True, draw_volume=True,
            spacing=spacing, color_map=color_map,
            scale_xy=scale_xy, scale_z=scale_z, max_z=max_z,
            type_ignore=ignore_types
        )
        return img
    except Exception as e:
        print(f"Diagram error: {e}")
        import traceback
        traceback.print_exc()
        return None

def classify_intent(query):
    # LOCAL keyword matching first — no API needed for obvious cases
    query_lower = query.lower()
    model_keywords = {
        "densenet": "DenseNet121", "resnet": "ResNet50", "vgg": "VGG16",
        "mobilenet": "MobileNetV2", "inception": "InceptionV3"
    }
    diagram_verbs = ["visualize", "draw", "diagram", "architecture", "show me", "generate", "render"]
    
    for key, model in model_keywords.items():
        if key in query_lower:
            return "diagram", model
    
    # Check for diagram-related verbs combined with "model" or "network"
    if any(v in query_lower for v in diagram_verbs) and any(w in query_lower for w in ["model", "network", "neural", "cnn", "net"]):
        return "diagram", "DenseNet121"
    
    # Fall back to Gemini router for ambiguous queries
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "chart", None
    prompt = f"""Classify into: "diagram" (3D neural net viz), "chart" (data analysis/plot), or "text" (markdown/table).
    If "diagram", extract model from ["DenseNet121","ResNet50","VGG16","MobileNetV2","InceptionV3"], default "DenseNet121".
    JSON only: {{"intent":"...","model_name":"..."}}
    Query: {query}"""
    try:
        client = genai.Client(api_key=api_key)
        r = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        t = r.text.replace('```json','').replace('```','').strip()
        d = json.loads(t)
        return d.get('intent','chart'), d.get('model_name')
    except:
        return "chart", None

def read_document_and_answer(query, file_path):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return "🚨 No Gemini API Key."
    try:
        if file_path.endswith(".pdf"):
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        if not text.strip(): return "🚨 No text extracted."
        text = text[:15000]
        prompt = f"""You are an expert academic assistant. Document text:
        ---
        {text}
        ---
        User asks: "{query}"
        Give a professional academic answer using markdown."""
        client = genai.Client(api_key=api_key)
        r = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return r.text
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# ============================================================
# REFERENCE HELPER
# ============================================================

def get_references(query, dataset_path="literature_dataset.json", max_refs=5):
    """Search for relevant references using ChromaDB vector search (fast) with JSON fallback."""
    # --- Fast path: ChromaDB semantic search ---
    if _chroma_collection is not None:
        try:
            query_embedding = _embedding_model.encode([query]).tolist()
            results = _chroma_collection.query(
                query_embeddings=query_embedding,
                n_results=max_refs * 2  # fetch extra to deduplicate by paper
            )
            if results and results['metadatas'] and results['metadatas'][0]:
                seen_titles = set()
                refs = "\n\n---\n**References:**\n"
                ref_count = 0
                for meta in results['metadatas'][0]:
                    title = meta.get('title', 'Untitled')
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    ref_count += 1
                    pmcid = meta.get('pmcid', '')
                    refs += f"[{ref_count}] \"{title}.\" (PMC{pmcid})\n"
                    if ref_count >= max_refs:
                        break
                if ref_count > 0:
                    return refs
        except Exception as e:
            print(f"ChromaDB search failed, falling back to JSON: {e}")

    # --- Fallback: keyword scan over JSON ---
    try:
        if not os.path.exists(dataset_path):
            return ""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 3]
        
        scored = []
        for paper in data:
            title = (paper.get('title', '') or '').lower()
            abstract = (paper.get('abstract', '') or '').lower()
            combined = title + ' ' + abstract
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scored.append((score, paper))
        
        scored.sort(key=lambda x: -x[0])
        top = scored[:max_refs]
        
        if not top:
            return ""
        
        refs = "\n\n---\n**References:**\n"
        for i, (_, paper) in enumerate(top, 1):
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                author_str = ', '.join(authors[:3])
                if len(authors) > 3:
                    author_str += ' et al.'
            else:
                author_str = str(authors)
            title = paper.get('title', 'Untitled')
            date = paper.get('date', 'n.d.')
            url = paper.get('url', '')
            year = date[:4] if date else 'n.d.'
            refs += f"[{i}] {author_str}. \"{title}.\" ({year})"
            if url:
                refs += f" {url}"
            refs += "\n"
        return refs
    except Exception:
        return ""

# ============================================================
# HYBRID THESIS GENERATOR (Qwen + Gemini → LaTeX + Figures)
# ============================================================

# Store generated figures per session for zip bundling
_thesis_figures = {}  # session_id -> list of (filename, filepath)

def detect_nn_model(topic):
    """Detect which NN model the topic refers to."""
    topic_lower = topic.lower()
    model_keywords = {
        "densenet": "DenseNet121", "resnet": "ResNet50", "vgg": "VGG16",
        "mobilenet": "MobileNetV2", "inception": "InceptionV3"
    }
    for key, model_name in model_keywords.items():
        if key in topic_lower:
            return model_name
    if any(w in topic_lower for w in ["x-ray", "xray", "chest", "medical", "radiology", "lung", "pneumonia"]):
        return "DenseNet121"
    return None

def generate_data_chart(topic):
    """Generate a relevant data chart from the literature dataset."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
        
    global _cached_literature_dataset
    if _cached_literature_dataset is not None:
        full_data = _cached_literature_dataset
    else:
        dataset_path = "literature_dataset.json"
        if not os.path.exists(dataset_path):
            return None
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
        except Exception:
            return None
            
    try:
        sample_data = full_data[:10]
        prompt = f"""You are an expert Data Scientist. Create a publication/research trend chart.
        Dataset: {len(full_data)} records. Sample: {json.dumps(sample_data, indent=2)}
        Topic context: "{topic}"
        Write Python that:
        1. Loads 'literature_dataset.json':
        `import json, pandas as pd`
        `with open('literature_dataset.json', 'r', encoding='utf-8') as f:`
        `    df = pd.DataFrame(json.load(f))`
        2. Creates a `fig` (Plotly figure) showing a relevant research trend.
        Chart styling: template="plotly_white", title_x=0.5, margin=dict(t=100,l=40,r=40,b=40).
        ONLY return raw Python code, no markdown backticks."""
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        code = response.text
        for prefix in ["```python", "```"]:
            if code.startswith(prefix): code = code[len(prefix):]
        if code.endswith("```"): code = code[:-3]
        code = code.strip()
        ns = globals().copy()
        exec(code, ns)
        if 'fig' in ns:
            chart_path = os.path.join(app.config['UPLOAD_FOLDER'], "literature_trends.png")
            ns['fig'].write_image(chart_path, width=900, height=500)
            return chart_path
    except Exception as e:
        print(f"Chart generation error: {e}")
    return None

def generate_thesis_stream(topic, user_name="User"):
    """Generator: Qwen drafts domain content → auto-generates figures → Gemini structures into LaTeX → SSE chunks."""
    import time, uuid

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield f"data: 🚨 No GEMINI_API_KEY found in .env file.\n\n"
        yield f"data: [DONE]\n\n"
        return

    # --- Step 0: Filter Conversational vs Thesis Intent ---
    try:
        client = genai.Client(api_key=api_key)
        
        # Hyper-fast intent check without streaming
        intent_prompt = f"""
        Does the following user input ask to write an academic paper, thesis, or complex research analysis?
        Input: "{topic}"
        
        If it's just asking for topic ideas, making casual conversation, or asking a generic question, output ONLY the word "CHAT".
        If it's explicitly asking to write a full thesis/paper, output ONLY the word "THESIS".
        """
        
        intent_res = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=intent_prompt
        ).text.strip().upper()
        
        if "THESIS" not in intent_res:
            # It's a conversational request. Stream the response directly to the user with zero buffering.
            chat_prompt = f"""
            You are a helpful AI research assistant named Acadbot. The user's name is {user_name}.
            They just said: "{topic}"
            
            Respond to them directly in a friendly, conversational manner. If they asked for ideas or topics, provide them naturally. Do not write a thesis.
            """
            
            response_stream = client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=chat_prompt
            )
            
            for chunk in response_stream:
                if chunk.text:
                    safe_text = chunk.text.replace('\n', '\\n')
                    yield f"data: {safe_text}\n\n"
                    
            yield f"data: [DONE]\n\n"
            return
            
    except Exception as e:
        print(f"Intent check failed: {e}")
        # Proceed to thesis generation if the check fails

    session_id = str(uuid.uuid4())[:8]
    figures = []

    # --- Step 1: Qwen domain research ---
    yield f"data: 📡 Step 1/5: Generating domain research using Qwen model...\n\n"
    time.sleep(0.1)
    try:
        qwen_draft = generate_draft(
            f"Write a detailed academic research analysis on: {topic}. "
            f"Include methodology, findings, and discussion relevant to chest X-ray image classification.",
            max_tokens=200, temperature=0.7
        )
    except Exception as e:
        qwen_draft = f"[Qwen model unavailable: {str(e)}. Proceeding with Gemini only.]"
    yield f"data: ✅ Domain research ready.\n\n"
    time.sleep(0.1)

    # --- Steps 2-4: Run in PARALLEL for speed ---
    yield f"data: 🚀 Steps 2-4: Fetching references, diagram, and chart in parallel...\n\n"

    detected_model = detect_nn_model(topic)

    def _fetch_refs():
        return get_references(topic)

    def _gen_diagram():
        if not detected_model:
            return None
        try:
            img = generate_model_diagram(detected_model)
            if img is not None:
                path = os.path.join(app.config['UPLOAD_FOLDER'], f"{detected_model}_architecture.png")
                img.save(path, format="PNG")
                return ("architecture.png", path, f"{detected_model} Architecture Diagram")
        except Exception as e:
            print(f"Diagram error: {e}")
        return None

    def _gen_chart():
        try:
            path = generate_data_chart(topic)
            if path and os.path.exists(path):
                return ("literature_trends.png", path, "Literature Research Trends")
        except Exception as e:
            print(f"Chart error: {e}")
        return None

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_refs = executor.submit(_fetch_refs)
        fut_diagram = executor.submit(_gen_diagram)
        fut_chart = executor.submit(_gen_chart)

    refs = fut_refs.result()
    ref_text = refs if refs else "No references found in the literature dataset."
    yield f"data: ✅ References collected.\n\n"

    diagram_result = fut_diagram.result()
    if diagram_result:
        figures.append(diagram_result)
        yield f"data: ✅ {detected_model} architecture diagram generated.\n\n"
    elif detected_model:
        yield f"data: ⚠️ Could not generate diagram for {detected_model}.\n\n"
    else:
        yield f"data: ℹ️ No specific NN model detected, skipping diagram.\n\n"

    chart_result = fut_chart.result()
    if chart_result:
        figures.append(chart_result)
        yield f"data: ✅ Literature trend chart generated.\n\n"
    else:
        yield f"data: ⚠️ Could not generate data chart.\n\n"

    time.sleep(0.1)

    # --- Step 5: Gemini LaTeX generation ---
    yield f"data: ✍️ Step 5/5: Generating complete LaTeX thesis via Gemini...\n\n"
    yield f"data: ---LATEX_START---\n\n"
    time.sleep(0.1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        yield f"data: 🚨 No GEMINI_API_KEY found in .env file.\n\n"
        yield f"data: [DONE]\n\n"
        return

    # Build figure instructions for Gemini
    figure_instructions = ""
    if figures:
        figure_instructions = "\n\nFIGURES TO INCLUDE (use \\includegraphics and \\figure environment):\n"
        for latex_name, _, caption in figures:
            figure_instructions += f"- Filename: {latex_name}, Caption: \"{caption}\"\n"
        figure_instructions += """
Use this exact pattern for each figure:
\\begin{figure}[htbp]
\\centering
\\includegraphics[width=\\columnwidth]{FILENAME}
\\caption{CAPTION}
\\label{fig:LABEL}
\\end{figure}

Add \\usepackage{graphicx} in the preamble.
Place the architecture diagram in the Methodology section.
Place the literature trends chart in the Related Work or Introduction section."""

    latex_prompt = f"""You are an expert academic researcher and LaTeX typesetter.

Using the domain-specific research content below (generated by a fine-tuned medical AI model),
write a COMPLETE IEEE-format LaTeX thesis/research paper on the topic: "{topic}"

DOMAIN RESEARCH CONTENT (from Qwen X-ray Researcher model):
---
{qwen_draft}
---

LITERATURE REFERENCES:
---
{ref_text}
---
{figure_instructions}

REQUIREMENTS:
1. Output ONLY valid LaTeX code, starting with \\documentclass and ending with \\end{{document}}
2. Use IEEE conference format (\\documentclass[conference]{{IEEEtran}})
3. Include these sections: Title, Abstract, Introduction, Related Work, Methodology, 
   Experimental Results, Discussion, Conclusion, References
4. Use the domain research content as the basis — rephrase and restructure it academically
5. Include proper \\cite{{}} commands for the references provided
6. Add \\bibliographystyle{{IEEEtran}} and \\begin{{thebibliography}} with the references
7. Make it publication-ready, 4-6 pages worth of content
8. Include \\usepackage{{graphicx}} for figures
9. Do NOT wrap in markdown code blocks, output raw LaTeX only

CRITICAL WRITING STYLE — You MUST follow ALL of these to sound like a real human researcher:

SENTENCE STRUCTURE (this is the most important part):
- NEVER write more than 2 consecutive sentences of similar length. Alternate between short (8-12 words) and long (25-40 words) sentences deliberately
- Start sentences with different words — NEVER begin 3+ consecutive sentences with "The", "This", "We", or "In"
- Use em-dashes occasionally — like this — to break up monotonous flow
- Include rhetorical questions sparingly ("But does this approach generalize to unseen data?")
- Use semicolons to join related ideas; this creates natural rhythm variation

VOICE AND TONE:
- Write in first-person plural throughout ("we propose", "our experiments show", "we hypothesize")
- Use academic hedging generously: "arguably", "it is worth noting", "one could argue", "to some extent", "it remains unclear whether"
- Show genuine uncertainty where appropriate ("while promising, these results warrant further investigation")
- Express mild surprise at unexpected findings ("Interestingly, the model performed better than anticipated")
- Reference your own limitations candidly ("We acknowledge that our dataset is relatively small")
- Occasionally use contractions where natural in academic context ("doesn't" instead of "does not" in informal discussion points — but sparingly)

PARAGRAPH FLOW:
- Vary paragraph lengths: 2-sentence paragraphs mixed with 6-7 sentence paragraphs
- Begin some paragraphs with a reference to a prior work, not a generic topic sentence
- End some paragraphs with a question or forward-looking statement instead of a conclusion
- Use transitional phrases that feel natural: "Building on this insight", "A closer look reveals", "This observation led us to", "Perhaps more importantly"
- Avoid formulaic transitions like "Furthermore", "Moreover", "Additionally" in consecutive paragraphs

CONTENT AUTHENTICITY:
- Include specific numerical details with realistic precision (e.g., "93.6\\%" not "93\\%" or "94\\%")
- Reference specific hyperparameters and training details naturally within sentences
- Add parenthetical asides: "(see Table 1)", "(trained on an NVIDIA RTX 3090 for approximately 12 hours)"
- Compare your results to baselines with nuanced analysis, not just "our method is better"
- Include at least one paragraph discussing unexpected results or challenges faced
- Mention practical implications ("This could reduce radiologist workload by...")"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=latex_prompt
        )
        for chunk in response:
            if chunk.text:
                safe_text = chunk.text.replace('\n', '\\n')
                yield f"data: {safe_text}\n\n"
    except Exception as e:
        yield f"data: 🚨 Gemini API error: {str(e)}\n\n"

    # Store figures for zip bundling
    _thesis_figures[session_id] = figures
    yield f"data: ---SESSION:{session_id}---\n\n"
    yield f"data: [DONE]\n\n"

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/thesis', methods=['POST'])
@login_required
def api_thesis():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 800)
    temperature = data.get('temperature', 0.7)
    result = generate_draft(prompt, max_tokens, temperature)
    refs = get_references(prompt)
    return jsonify({"response": result + refs})

@app.route('/api/thesis/stream')
@login_required
def api_thesis_stream():
    """SSE endpoint: streams LaTeX thesis generation word-by-word."""
    topic = request.args.get('topic', '')
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    user_name = getattr(request, 'user', {}).get('name', 'User')
    return Response(
        generate_thesis_stream(topic, user_name),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

@app.route('/api/thesis/download', methods=['POST'])
@login_required
def api_thesis_download():
    """Bundle LaTeX + figures into a .zip and return for download."""
    import zipfile
    data = request.json
    latex_content = data.get('latex', '')
    session_id = data.get('session_id', '')
    filename = data.get('filename', 'thesis')

    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{filename}/{filename}.tex", latex_content)
        figures = _thesis_figures.get(session_id, [])
        for latex_name, abs_path, _ in figures:
            if os.path.exists(abs_path):
                zf.write(abs_path, f"{filename}/{latex_name}")

    return send_file(zip_path, as_attachment=True, download_name=f"{filename}.zip", mimetype='application/zip')

@app.route('/api/visualize', methods=['POST'])
@login_required
def api_visualize():
    prompt = request.form.get('prompt', '')
    file = request.files.get('file')
    dataset_path = "literature_dataset.json"
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        dataset_path = path
    
    intent, model_name = classify_intent(prompt)
    print(f"[Router] intent={intent}, model_name={model_name}, prompt={prompt}")
    
    if intent == "diagram":
        target = model_name or "DenseNet121"
        try:
            img = generate_model_diagram(target)
            if img is not None:
                tmp = os.path.join(app.config['UPLOAD_FOLDER'], f"{target}_diagram.png")
                img.save(tmp, format="PNG")
                print(f"Diagram saved to {tmp}")
                return jsonify({"type": "image", "url": f"/api/image/{os.path.basename(tmp)}", "caption": f"{target} Architecture", "references": get_references(f"{target} neural network architecture deep learning")})
            else:
                return jsonify({"type": "text", "response": "Failed to instantiate model. Please try again."})
        except Exception as e:
            print(f"Diagram API error: {e}")
            return jsonify({"type": "text", "response": f"Error generating diagram: {str(e)}"})
    else:
        fig, md = analyze_data(prompt, dataset_path=dataset_path)
        if fig is not None:
            tmp = os.path.join(app.config['UPLOAD_FOLDER'], "chart.png")
            fig.write_image(tmp, width=900, height=500)
            return jsonify({"type": "image", "url": f"/api/image/chart.png", "caption": "Generated Chart", "references": get_references(prompt)})
        elif md:
            refs = get_references(prompt)
            return jsonify({"type": "text", "response": md + refs})
        else:
            return jsonify({"type": "text", "response": "🚨 No output generated."})

@app.route('/api/document', methods=['POST'])
@login_required
def api_document():
    prompt = request.form.get('prompt', '')
    file = request.files.get('file')
    doc_path = request.form.get('doc_path', '')
    
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        return jsonify({"response": f"✅ Document **{file.filename}** uploaded successfully! Ask me anything about it.", "doc_path": path})
    
    if not doc_path or doc_path == 'undefined' or doc_path == 'null':
        # Fallback: Instead of an error, query the built-in literature dataset
        dataset_refs = get_references(prompt, max_refs=8)
        if not dataset_refs:
            return jsonify({"response": "I couldn't find any relevant information in the medical literature dataset."})
        
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            sys_prompt = f"""You are an expert academic assistant. Use ONLY these literature references to answer the query.
            {dataset_refs}
            
            User asks: "{prompt}"
            Give a professional academic answer using markdown."""
            
            r = client.models.generate_content(model='gemini-2.5-flash', contents=sys_prompt)
            return jsonify({"response": r.text + dataset_refs})
        except Exception as e:
            return jsonify({"response": f"🚨 Error analyzing dataset: {str(e)}"})
    
    result = read_document_and_answer(prompt, doc_path)
    return jsonify({"response": result, "doc_path": doc_path})

@app.route('/api/image/<filename>')
def serve_image(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[Image] Serving: {path}, exists={os.path.exists(path)}")
    return send_file(path, mimetype='image/png')

@app.route('/api/test_diagram/<model_name>')
def test_diagram(model_name):
    """Direct test endpoint — bypasses the router entirely."""
    try:
        img = generate_model_diagram(model_name)
        if img is not None:
            tmp = os.path.join(app.config['UPLOAD_FOLDER'], f"{model_name}_test.png")
            img.save(tmp, format="PNG")
            return send_file(tmp, mimetype='image/png')
        else:
            return "generate_model_diagram returned None", 500
    except Exception as e:
        return f"Exception: {str(e)}", 500

# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Pre-load the text model at startup to eliminate cold-start latency
    print("Pre-loading text model at startup...")
    success, msg = load_text_model()
    print(f"Text model: {msg}")

    print("Launching Acadbot on http://localhost:5000")
    app.run(debug=False, port=5000)
