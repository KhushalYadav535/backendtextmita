import time
from huggingface_hub import snapshot_download
import urllib3
import requests

def robust_hf_download(repo_id, local_dir, max_retries=50):
    for attempt in range(max_retries):
        try:
            print(f"Starting/resuming HuggingFace download... (Attempt {attempt + 1}/{max_retries})")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
            print("Download completed successfully!")
            break
        except (urllib3.exceptions.ProtocolError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError, Exception) as e:
            print(f"Connection dropped: {e}")
            print("Reconnecting in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    robust_hf_download('opendatalab/PDF-Extract-Kit', 'C:\\Users\\khush\\.models')
