import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_model(slug, is_dataset=False, download_path="./trained_model"):
    """
    Downloads the trained model weights from Kaggle.
    Requires kaggle API credentials (kaggle.json) present in ~/.kaggle/
    """
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Failed to authenticate with Kaggle API. Please ensure you have your kaggle.json in ~/.kaggle/")
        print(f"Error: {e}")
        return

    os.makedirs(download_path, exist_ok=True)

    if is_dataset:
        print(f"Downloading dataset {slug} into {download_path}...")
        # unzip=True automatically extracts the zip file downloaded from Kaggle
        api.dataset_download_files(slug, path=download_path, unzip=True)
    else:
        print(f"Downloading notebook output from {slug} into {download_path}...")
        # This will download the notebook output files. 
        # If it downloads a zip, it will be placed in the directory.
        api.kernels_output(slug, path=download_path)
    
    print(f"Done! Files saved to {os.path.abspath(download_path)}")

if __name__ == "__main__":
    # --- CONFIGURE THESE VARIABLES ---
    # Replace with your Kaggle username and the notebook/dataset name
    # Example Notebook Slug: "yourusername/qwen-lora-finetuning"
    # Example Dataset Slug: "yourusername/my-fine-tuned-model"
    KAGGLE_SLUG = "your-kaggle-username/your-notebook-or-dataset-slug"
    
    # Set to True if you uploaded the saved weights to Kaggle as a Dataset.
    # Set to False if you just ran a Notebook and want to download its /kaggle/working/ output.
    IS_DATASET = False 
    
    download_path = "./trained_model"
    
    download_kaggle_model(KAGGLE_SLUG, is_dataset=IS_DATASET, download_path=download_path)
