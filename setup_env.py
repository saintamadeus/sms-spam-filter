import os
import urllib.request
import zipfile
import csv

# Create folders
folders = ['data', 'notebooks', 'src', 'models', 'screenshots']
for f in folders:
    os.makedirs(f, exist_ok=True)
    print(f"Created folder: {f}")

# Download dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "data/sms_spam.zip"
print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data/")

# Convert SMSSpamCollection (TSV) to CSV matching Kaggle format
txt_path = "data/SMSSpamCollection"
csv_path = "data/spam.csv"

print("Converting to CSV format...")
with open(txt_path, 'r', encoding='utf-8') as f_in, open(csv_path, 'w', newline='', encoding='latin-1', errors='replace') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['v1', 'v2']) # Headers
    for line in f_in:
        parts = line.strip('\n').split('\t')
        if len(parts) >= 2:
            writer.writerow([parts[0], parts[1]])

# Clean up
os.remove(zip_path)
os.remove(txt_path)
if os.path.exists("data/readme"):
    os.remove("data/readme")

print("Setup completed successfully.")
