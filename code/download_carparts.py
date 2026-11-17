import urllib.request
import zipfile
import os
import pandas as pd
from io import BytesIO

url = "https://zenodo.org/records/4656014/files/car_parts_dataset_without_missing_values.zip"
print(f"Downloading {url}...")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            z.extractall("car_parts_data")
            print("Extracted to car_parts_data/")
            
    # Check the format of the extracted data
    files = os.listdir("car_parts_data")
    print(f"Files: {files}")
    
    # Typically Monash format is a .tsf file
    # Let's see if we can parse it
    
except Exception as e:
    print(f"Error downloading: {e}")
