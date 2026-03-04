import os
import requests
import xml.etree.ElementTree as ET
import time
import json

def fetch_pmc_full_text():
    print("Querying PubMed Central (PMC) for Open Access Full-Text Papers...")
    
    # E-utilities search endpoint targeting deep learning and chest x-rays with open access
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pmc",
        "term": '("chest x-ray" OR pneumonia) AND "deep learning" AND "open access"[filter]',
        "retmode": "json",
        "retmax": 20 # Starting with 20 papers for testing the RAG Database
    }
    
    response = requests.get(search_url, params=search_params)
    response.raise_for_status()
    pmc_ids = response.json().get("esearchresult", {}).get("idlist", [])
    
    print(f"Found {len(pmc_ids)} papers. Downloading full-text XML and extracting images...")
    
    # Create directories for the RAG database and image storage
    os.makedirs("pmc_data_rag", exist_ok=True)
    os.makedirs(os.path.join("pmc_data_rag", "images"), exist_ok=True)
    
    dataset = []
    
    for pmcid in pmc_ids:
        print(f"Fetching PMC{pmcid}...")
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
        
        try:
            xml_resp = requests.get(fetch_url)
            xml_resp.raise_for_status()
            
            # Parse the XML
            root = ET.fromstring(xml_resp.content)
            
            # Extract Title
            title_node = root.find(".//article-title")
            title = "".join(title_node.itertext()) if title_node is not None else "Unknown Title"
            
            # Extract Abstract
            abstract_texts = root.findall(".//abstract//p")
            abstract = " ".join(["".join(p.itertext()) for p in abstract_texts])
            
            # Extract Body Text
            body_texts = root.findall(".//body//p")
            body = " ".join(["".join(p.itertext()) for p in body_texts])
            
            full_text = f"Title: {title}\nAbstract: {abstract}\nBody: {body}"
            
            # Extract Figures and Download Images
            figures = root.findall(".//fig")
            extracted_images = []
            
            for fig in figures:
                caption_node = fig.find(".//caption")
                caption = "".join(caption_node.itertext()).strip() if caption_node is not None else ""
                
                graphic = fig.find(".//graphic")
                if graphic is not None:
                    href = None
                    for key, val in graphic.attrib.items():
                        if "href" in key:
                            href = val
                            break
                    
                    if href:
                        # PMC Base URL for retrieving the linked images
                        img_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/bin/{href}.jpg"
                        img_filename = f"PMC{pmcid}_{href}.jpg"
                        img_path = os.path.join("pmc_data_rag", "images", img_filename)
                        
                        try:
                            # Download the actual image file to the local directory
                            img_data = requests.get(img_url).content
                            with open(img_path, 'wb') as handler:
                                handler.write(img_data)
                                
                            extracted_images.append({
                                "image_path": img_path,
                                "caption": caption
                            })
                            print(f"  -> Extracted Image: {img_filename}")
                        except Exception as img_e:
                            print(f"  Could not download image {href}: {img_e}")
            
            paper_data = {
                "pmcid": pmcid,
                "title": title,
                "text_content": full_text,
                "images": extracted_images
            }
            dataset.append(paper_data)
            
            # Polite rate limiting for NCBI API
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error processing PMC{pmcid}: {e}")
            
    # Save the final structured vector database input
    db_path = os.path.join("pmc_data_rag", "full_text_database.json")
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
        
    print(f"\nPhase 2 Download Complete! Saved {len(dataset)} full-text papers and extracted images to './pmc_data_rag'.")

if __name__ == "__main__":
    fetch_pmc_full_text()
