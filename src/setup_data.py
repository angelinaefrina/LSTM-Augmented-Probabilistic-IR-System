import os
import json
import urllib.request

def download_file(url, filepath):
    print(f"Downloading {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print("Done.")

def setup_cranfield():
    target_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(target_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/aliahmedghazi/Vector_Space_Cranfield_Dataset/master/"
    
    files = {
        "cran_docs.json": base_url + "cranfield_data.json",
        "cran_queries.json": base_url + "cran.qry.json",
        "cran_qrels.json": base_url + "cranqrel.json"
    }

    for filename, url in files.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            download_file(url, filepath)
        else:
            print(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    print("Initializing Data Setup...")
    setup_cranfield()
    print("Data setup complete. Files are located in data/raw/")