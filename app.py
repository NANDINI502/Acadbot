import os
import torch
import json
import tempfile
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

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
TEXT_MODEL_DIR = "./qwen-xray-researcher"
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Globals ---
text_tokenizer = None
text_model = None

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
        text_model = AutoPeftModelForCausalLM.from_pretrained(
            TEXT_MODEL_DIR, 
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True
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
        generated_ids = text_model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=temperature,
            do_sample=True, repetition_penalty=1.1
        )
    input_len = inputs["input_ids"].shape[1]
    generated_text = text_tokenizer.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
    return generated_text.split("Keywords:")[0].strip()

def analyze_data(query, dataset_path="literature_dataset.json"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None, "🚨 No Gemini API Key found in .env"
    if not os.path.exists(dataset_path): return None, f"🚨 '{dataset_path}' not found."
    try:
        if dataset_path.endswith(".csv"):
            full_data = pd.read_csv(dataset_path).to_dict(orient="records")
        else:
            with open(dataset_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
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
        model = model_map[model_name](weights=None, include_top=True)
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
        
        # Monkey-patch: TF 2.20 removed output_shape from InputLayer
        for layer in model.layers:
            if not hasattr(layer, 'output_shape'):
                try:
                    layer.output_shape = layer.output.shape
                except Exception:
                    layer.output_shape = (None, 224, 224, 3)
        
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
    """Search the literature dataset for papers relevant to the query."""
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
    dataset_path = "literature_dataset.json"
    if not os.path.exists(dataset_path):
        return None
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
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

def generate_thesis_stream(topic):
    """Generator: Qwen drafts domain content → auto-generates figures → Gemini structures into LaTeX → SSE chunks."""
    import time, uuid

    session_id = str(uuid.uuid4())[:8]
    figures = []

    # --- Step 1: Qwen domain research ---
    yield f"data: 📡 Step 1/5: Generating domain research using Qwen model...\n\n"
    time.sleep(0.1)
    try:
        qwen_draft = generate_draft(
            f"Write a detailed academic research analysis on: {topic}. "
            f"Include methodology, findings, and discussion relevant to chest X-ray image classification.",
            max_tokens=800, temperature=0.7
        )
    except Exception as e:
        qwen_draft = f"[Qwen model unavailable: {str(e)}. Proceeding with Gemini only.]"
    yield f"data: ✅ Domain research ready.\n\n"
    time.sleep(0.1)

    # --- Step 2: Literature references ---
    yield f"data: 📚 Step 2/5: Fetching literature references...\n\n"
    refs = get_references(topic)
    ref_text = refs if refs else "No references found in the literature dataset."
    yield f"data: ✅ References collected.\n\n"
    time.sleep(0.1)

    # --- Step 3: NN architecture diagram ---
    yield f"data: 🧠 Step 3/5: Generating neural network architecture diagram...\n\n"
    detected_model = detect_nn_model(topic)
    if detected_model:
        try:
            img = generate_model_diagram(detected_model)
            if img is not None:
                diagram_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{detected_model}_architecture.png")
                img.save(diagram_path, format="PNG")
                figures.append(("architecture.png", diagram_path, f"{detected_model} Architecture Diagram"))
                yield f"data: ✅ {detected_model} architecture diagram generated.\n\n"
            else:
                yield f"data: ⚠️ Could not generate diagram for {detected_model}.\n\n"
        except Exception as e:
            yield f"data: ⚠️ Diagram error: {str(e)}\n\n"
    else:
        yield f"data: ℹ️ No specific NN model detected, skipping diagram.\n\n"
    time.sleep(0.1)

    # --- Step 4: Data chart ---
    yield f"data: 📊 Step 4/5: Generating literature trend chart...\n\n"
    try:
        chart_path = generate_data_chart(topic)
        if chart_path and os.path.exists(chart_path):
            figures.append(("literature_trends.png", chart_path, "Literature Research Trends"))
            yield f"data: ✅ Literature trend chart generated.\n\n"
        else:
            yield f"data: ⚠️ Could not generate data chart.\n\n"
    except Exception as e:
        yield f"data: ⚠️ Chart error: {str(e)}\n\n"
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
def api_thesis():
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 800)
    temperature = data.get('temperature', 0.7)
    result = generate_draft(prompt, max_tokens, temperature)
    refs = get_references(prompt)
    return jsonify({"response": result + refs})

@app.route('/api/thesis/stream')
def api_thesis_stream():
    """SSE endpoint: streams LaTeX thesis generation word-by-word."""
    topic = request.args.get('topic', '')
    if not topic:
        return jsonify({"error": "No topic provided"}), 400
    return Response(
        generate_thesis_stream(topic),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

@app.route('/api/thesis/download', methods=['POST'])
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
def api_document():
    prompt = request.form.get('prompt', '')
    file = request.files.get('file')
    doc_path = request.form.get('doc_path', '')
    
    if file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        return jsonify({"response": f"✅ Document **{file.filename}** uploaded successfully! Ask me anything about it.", "doc_path": path})
    
    if not doc_path:
        return jsonify({"response": "🚨 Please upload a document first."})
    
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
    print("Launching Acadbot on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
