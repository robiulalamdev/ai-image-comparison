import os
import gdown

# Google Drive shareable file link
drive_link = "https://drive.google.com/file/d/1_kXSse6uMIDNTx0b-i5HMJeEhqLtLQ-1/view?usp=sharing"

# Extract file ID from the URL
def extract_file_id(url):
    parts = url.split('/')
    if 'file' in parts and 'd' in parts:
        try:
            return parts[5]
        except IndexError:
            raise ValueError("Invalid Google Drive link format.")
    raise ValueError("Invalid Google Drive link format.")

# Download function
def download_file_from_drive(link, output_dir):
    file_id = extract_file_id(link)
    output_path = os.path.join(output_dir, 'changeformer_levir.pth')  # You can customize this
    gdown.download(id=file_id, output=output_path, quiet=False)
    print(f"✅ File saved to {output_path}")

# Create output directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

# Run download
download_file_from_drive(drive_link, "checkpoints")
