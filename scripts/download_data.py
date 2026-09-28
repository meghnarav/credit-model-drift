import kagglehub
import shutil
import os
import glob

def download_heloc():
    print("Downloading HELOC dataset using kagglehub...")
    path = kagglehub.dataset_download("averkiyoliabev/home-equity-line-of-creditheloc")
    print("Downloaded to:", path)
    
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("Could not find any CSV files in the downloaded dataset.")
        
    src_csv = csv_files[0]
    dest_dir = "data/raw"
    os.makedirs(dest_dir, exist_ok=True)
    dest_csv = os.path.join(dest_dir, "heloc_dataset.csv")
    
    shutil.copy2(src_csv, dest_csv)
    print(f"Successfully copied {src_csv} to {dest_csv}")

if __name__ == "__main__":
    download_heloc()
