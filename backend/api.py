from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile
from inference import get_latest_model, setup_predictor, process_pdf
from extraction import process_extracted_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model_path = get_latest_model()
model = None
try:
    if model_path and model_path.exists():
        model = setup_predictor(str(model_path))
except Exception as e:
    print(f"Warning: Could not load model: {e}")

@app.post("/upload")
async def extract_data(files: list[UploadFile] = File(...)):
    results = {}
    if not model:
        return {"error": "Model not loaded properly or missing"}
    
    for idx, file in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing: {file.filename}...")
        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(temp_fd)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Run inference per pdf
            file_results = process_pdf(temp_path, model, original_filename=file.filename)
            processed_results = process_extracted_data(file_results)
            results[file.filename] = processed_results
            print(f"      - Found {len(file_results)} regions in {file.filename}")
        except Exception as e:
            print(f"      - Error processing {file.filename}: {e}")
            results[file.filename] = {"error": str(e)}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
