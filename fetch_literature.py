import os
import time
import json
try:
    import arxiv
except ImportError:
    print("Installing the required 'arxiv' package to query academic papers...")
    os.system("pip install arxiv")
    import arxiv

def fetch_literature():
    print("Querying the ArXiv Database for Academic Papers...")
    print("Topic: Deep Learning & AI Classification of Chest X-Rays\n")
    
    # We will search for broader medical imaging and deep learning papers to hit the 10k target
    search_query = '(all:"chest x-ray" OR all:"pneumonia" OR all:"medical imaging" OR all:"radiology") AND (all:"deep learning" OR all:"AI" OR all:"CNN" OR all:"neural network")'
    
    # We use the official arxiv Python client to make requests cleanly
    client = arxiv.Client(page_size=1000, delay_seconds=3, num_retries=3) # Add explicit rate limit protection for massive scale
    search = arxiv.Search(
        query=search_query,
        max_results=10000, # Grabbing 10,000 matching papers
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    dataset = []
    count = 0
    
    # Create the generator to iterate over results
    results = client.results(search)
    
    try:
        for result in results:
            count += 1
            
            # Clean up the raw strings (removing newlines so it formats well for AI)
            title = result.title.replace('\n', ' ')
            abstract = result.summary.replace('\n', ' ')
            authors = [author.name for author in result.authors]
            published = result.published.strftime("%Y-%m-%d")
            
            print(f"[{count}/10000] Downloading: {title[:70]}...")
            
            # Structure into a dictionary
            paper_data = {
                "id": count,
                "title": title,
                "authors": authors,
                "date": published,
                "url": result.entry_id,
                "abstract": abstract
            }
            
            dataset.append(paper_data)
            
            # Sleep slightly to prevent hammering the ArXiv API rate limits during massive pulls
            if count % 50 == 0:
                time.sleep(2)
                
    except Exception as e:
        print(f"\nEncountered an error while fetching: {e}")
        print(f"Successfully downloaded {count} papers before the error.")

    # Save to a highly structured JSON File (JSONL is best for LLM Training datasets)
    output_file = "literature_dataset.json"
    print(f"\nFinished! Saving {len(dataset)} papers to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    fetch_literature()
