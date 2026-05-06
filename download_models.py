import os
import tarfile
import urllib.request
import urllib.error
import time

def download_with_retry(url, dest_path, retries=10):
    for i in range(retries):
        try:
            print(f"Downloading {url} (Attempt {i+1}/{retries})...")
            
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (chunk_size * 100) == 0:
                            print(f"\rProgress: {downloaded}/{total_size} bytes ({(downloaded/total_size)*100:.1f}%)", end='')
                
            print(f"\nSuccessfully downloaded to {dest_path}")
            return True
            
        except Exception as e:
            print(f"\nDownload failed: {e}")
            time.sleep(2)
            
    return False

def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)

def setup_paddle_models():
    base_dir = os.path.expanduser("~/.paddleocr/whl")
    
    models = [
        {
            "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar",
            "dir": os.path.join(base_dir, "rec", "en", "en_PP-OCRv4_rec_infer")
        },
        {
            "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
            "dir": os.path.join(base_dir, "cls", "ch_ppocr_mobile_v2.0_cls_infer")
        }
    ]
    
    for model in models:
        os.makedirs(model["dir"], exist_ok=True)
        tar_path = os.path.join(model["dir"], os.path.basename(model["url"]))
        
        success = download_with_retry(model["url"], tar_path)
        if success:
            print(f"Extracting {tar_path}...")
            with tarfile.open(tar_path, 'r') as tar:
                safe_extract(tar, os.path.dirname(model["dir"]))
            print("Extraction complete.")

if __name__ == "__main__":
    setup_paddle_models()
    print("Done setting up models!")
