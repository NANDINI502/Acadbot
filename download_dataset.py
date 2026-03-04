import os
try:
    from datasets import load_dataset
    from PIL import Image
except ImportError:
    print("Installing required libraries: datasets and pillow...")
    os.system("pip install datasets pillow")
    from datasets import load_dataset
    from PIL import Image

def download_and_explore_dataset():
    print("Downloading the Chest X-Ray (Pneumonia) dataset from Hugging Face...")
    print("This might take a few minutes depending on your internet connection (approx 1.2 GB).")
    
    # We use a popular, highly-rated Chest X-Ray dataset available on Hugging Face
    # It contains over 5,800 X-Ray images classified as 'Normal' or 'Pneumonia'
    # We use a standard parquet-based Chest X-Ray dataset to avoid script execution errors
    dataset = load_dataset("hf-vision/chest-xray-pneumonia")
    
    print("\nDataset successfully downloaded and loaded into memory!")
    print("\n--- Dataset Overview ---")
    print(dataset)
    
    # Let's inspect the first training example
    print("\n--- First Training Example ---")
    sample = dataset['train'][0]
    # Datasets often have 'label' or 'labels'
    label_key = 'labels' if 'labels' in sample else 'label' if 'label' in sample else None
    
    print(f"Sample data keys: {sample.keys()}")
    if label_key:
        print(f"Label: {sample[label_key]}")
    
    # Create a local directory to save some sample images
    os.makedirs("sample_images", exist_ok=True)
    
    # Save a few sample images locally so you can view them
    print("\nSaving a few sample images to the 'sample_images' folder...")
    for i in range(3):
        image = dataset['train'][i]['image']
        
        # Determine the label name
        if label_key:
            label_val = dataset['train'][i][label_key]
            label_name = f"class_{label_val}"
        else:
            label_name = "unknown"
            
        image_path = f"sample_images/sample_{i}_{label_name}.jpeg"
        # Convert to RGB in case it's in a different format and save
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)
        print(f"Saved: {image_path}")

if __name__ == "__main__":
    download_and_explore_dataset()
    print("\nDone! You are ready to start pre-processing the data for your SLM.")
