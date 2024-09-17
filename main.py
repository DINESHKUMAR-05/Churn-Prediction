from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from model import CustomPipeline
import pandas as pd
from io import StringIO
import os

app = FastAPI()

pipeline = CustomPipeline()

@app.post("/train_model/")
async def train_model(file: UploadFile = File(...)):
    contents = await file.read()
    data = pd.read_csv(StringIO(contents.decode('utf-8')))

    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)

    pipeline.train_model(X_train, y_train)

    evaluation_results = pipeline.evaluate_model(X_test, y_test)
    
    model_file_path = pipeline.save_best_model()

    return {
        "message": "Model trained successfully",
        "evaluation_results": evaluation_results,
        "model_file": model_file_path
    }

@app.get("/download_model/")
def download_model():
    model_file_path = "best_model.pkl"
    if os.path.exists(model_file_path):
        return FileResponse(path=model_file_path, filename="best_model.pkl", media_type='application/octet-stream')
    else:
        return {"error": "Model file not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)