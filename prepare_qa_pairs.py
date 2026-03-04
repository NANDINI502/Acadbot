import json
import random

def format_conversational_dataset():
    input_file = "literature_dataset.json"
    output_file = "research_qa_dataset.jsonl"
    
    print(f"Loading raw literature from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print(f"Could not find {input_file}. You must run fetch_literature.py first!")
        return
        
    print(f"Found {len(papers)} papers! Processing into Q&A format...")
    
    # We will generate synthetic instruction-response pairs
    PROMPT_TEMPLATES = [
        "What are the findings of {title}?",
        "Summarize the research presented in {title}.",
        "Explain the abstract of the study titled '{title}'.",
        "Can you provide an overview of the paper {title} by {authors}?",
        "Discuss the medical AI research from the paper '{title}'."
    ]

    structured_dataset = []
    
    for paper in papers:
        # 1. Pick a random prompt style
        template = random.choice(PROMPT_TEMPLATES)
        
        # 2. Format the prompt with the paper's actual data
        authors_str = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            authors_str += " et al."
            
        instruction = template.format(
            title=paper["title"], 
            authors=authors_str
        )
        
        # 3. Format the ideal response that the SLM should learn to generate
        response = f"The study '{paper['title']}' was published on {paper['date']}. It details the following research:\n\n{paper['abstract']}"
        
        # 4. Save in the Hugging Face standard 'messages' format for Chat / Instruct models
        sft_data = {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]
        }
        structured_dataset.append(sft_data)
        
    # Write to a JSONL file (JSON Lines - standard for SLM fine-tuning)
    print(f"Saving formatted dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in structured_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Success! Generated {len(structured_dataset)} high-quality Q&A samples for NLP fine-tuning.")

if __name__ == "__main__":
    format_conversational_dataset()
