import os
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import numpy as np # type: ignore
import zipfile
from fastapi import FastAPI
import app.defaker_backend as df

app = FastAPI()

# =================================================================================
#                         Helper Functions
# =================================================================================


#@app.get("/endpoint/template")
#async def api_function_template():
#    return "This is an example of an API endpoint"

def setup_kaggle_api():
    """
    Initialize and authenticate Kaggle API
    Returns:
        KaggleApi: Authenticated Kaggle API instance
    """
    api = KaggleApi()
    api.authenticate()
    return api

def download_van_gogh_dataset(api, download_path='./kaggle/input/van-gogh-paintings/Paris'):
    """
    Download the Van Gogh paintings dataset using Kaggle API
    
    Args:
        api (KaggleApi): Authenticated Kaggle API instance
        download_path (str): Path where to download the dataset
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Download dataset
    dataset_name = "ipythonx/van-gogh-paintings"
    print(f"Downloading dataset from {dataset_name}...")
    
    try:
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded successfully!")
    except Exception as e:
        raise Exception(f"Error downloading dataset: {str(e)}")

def load_images_to_array(download_path='./kaggle/input/van-gogh-paintings/Paris'):
    """
    Load downloaded images into numpy array
    
    Args:
        download_path (str): Path where dataset is downloaded
        
    Returns:
        list: List of numpy arrays containing images
        list: List of corresponding image filenames
    """
    kaggle_img_array = []
    image_filenames = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Walk through all directories in the downloaded dataset
    for root, _, files in os.walk(download_path):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                try:
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    
                    kaggle_img_array.append(img_array)
                    image_filenames.append(filename)
                    
                    print(f"Loaded image: {filename}")
                except Exception as e:
                    print(f"Error loading image {filename}: {str(e)}")
    
    return kaggle_img_array, image_filenames

def get_van_gogh_paintings():
    """
    Main function to download and load Van Gogh paintings
    
    Returns:
        tuple: (list of image arrays, list of image filenames)
    """
    try:
        # Setup API
        api = setup_kaggle_api()
        
        # Set download path
        download_path = './van-gogh-dataset'
        
        # Download dataset if not already present
        if not os.path.exists(download_path) or not os.listdir(download_path):
            download_van_gogh_dataset(api, download_path)
        
        # Load images
        images, filenames = load_images_to_array(download_path)
        
        print(f"\nSuccessfully loaded {len(images)} images")
        return images, filenames
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return [], []


if __name__ == "__main__":
    
    image_arrays, image_names = get_van_gogh_paintings()
    
    # Example of accessing the loaded images
    if image_arrays:
        print("\nFirst few images loaded:")
        for i in range(min(3, len(image_arrays))):
            print(f"Image: {image_names[i]}")
            print(f"Shape: {image_arrays[i].shape}")
            print(f"Data type: {image_arrays[i].dtype}")
            print("---")

# =================================================================================
#                           Endpoints for GAN classes
# =================================================================================
