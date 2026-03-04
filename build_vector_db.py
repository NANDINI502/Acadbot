import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model for creating vector embeddings
print("Loading embedding model (this may download model weights the first time)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_vector_database():
    print("Building Vector Database from PMC Full-Text Data...")
    
    # Check if the data exists
    db_path = os.path.join("pmc_data_rag", "full_text_database.json")
    if not os.path.exists(db_path):
        print(f"Error: Could not find {db_path}. Please run fetch_full_text.py first.")
        return
        
    with open(db_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} full-text papers.")
    
    # Initialize ChromaDB client (stores DB locally in the 'chroma_db' folder)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get the collection
    collection = chroma_client.get_or_create_collection(
        name="medical_literature",
        metadata={"hnsw:space": "cosine"} # Use cosine similarity for text search
    )
    
    documents = []
    metadatas = []
    ids = []
    
    doc_id_counter = 0
    
    for paper in dataset:
        pmcid = paper.get("pmcid", "")
        title = paper.get("title", "")
        text_content = paper.get("text_content", "")
        images = paper.get("images", [])
        
        # We need to chunk the massive text so the Vector DB can search it accurately
        # A simple chunking by paragraphs (splitting by double newline or taking 1000 character chunks)
        # For simplicity, we split by ~500 character chunks
        chunk_size = 500
        text_chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
        
        # Format the images as a JSON string to store in metadata
        images_json = json.dumps(images)
        
        for i, chunk in enumerate(text_chunks):
            # Only add meaningful chunks
            if len(chunk.strip()) > 50:
                doc_id = f"PMC{pmcid}_chunk{i}"
                documents.append(chunk)
                
                # Metadata stores the original paper info AND the image links
                metadatas.append({
                    "pmcid": str(pmcid),
                    "title": title,
                    "images_json": images_json
                })
                
                ids.append(doc_id)
                doc_id_counter += 1
                
    print(f"Created {len(documents)} text chunks. Generating vector embeddings and inserting into ChromaDB...")
    print("This might take a minute depending on CPU...")
    
    # Generate embeddings in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        
        # Calculate embeddings
        batch_embeddings = embedding_model.encode(batch_docs).tolist()
        
        # Add to ChromaDB
        collection.upsert(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        
        if (i+1) % 500 == 0 or (i + batch_size) >= len(documents):
            print(f"  -> Processed {min(i+batch_size, len(documents))}/{len(documents)} chunks...")
            
    print("\nVector Database successfully built and saved locally to './chroma_db'!")
    print("The custom AI agent will now use this database to retrieve factual medical literature and accurate reference images.")

if __name__ == "__main__":
    build_vector_database()
