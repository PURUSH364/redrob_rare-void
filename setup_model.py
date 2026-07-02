from sentence_transformers import SentenceTransformer

def download_model():
    model_name = "all-MiniLM-L6-v2"
    local_dir = "./local_model"
    
    print(f"Downloading {model_name} to {local_dir}...")
    model = SentenceTransformer(model_name)
    model.save(local_dir)
    print("Download and save complete. Ready for offline use.")

if __name__ == "__main__":
    download_model()
