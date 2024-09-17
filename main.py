import io
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from model import CustomPipeline

app = FastAPI()

class ModelOutput(BaseModel):
    model_name: str
    accuracy: float
    classification_report: str
    feature_importances: dict
    pdp_plots: list
    shap_summary_plot: str
    lime_explanation: str
    surrogate_tree: str

@app.post("/process_dataset/", response_model=ModelOutput)
async def process_dataset(file: UploadFile = File(...)):
    content = await file.read()
    data = pd.read_csv(io.BytesIO(content))
    
    pipeline = CustomPipeline()
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    
    best_model_name, best_model = pipeline.train_model(X_train, y_train)
    accuracy, report = pipeline.evaluate_model(best_model, X_test, y_test)
    
    feature_importances = pipeline.get_feature_importances(best_model)
    pdp_plots = pipeline.generate_pdp_plots(best_model, X_test)
    shap_summary_plot = pipeline.generate_shap_summary_plot(best_model, X_train, X_test)
    lime_explanation = pipeline.generate_lime_explanation(best_model, X_train, X_test)
    surrogate_tree = pipeline.generate_surrogate_tree(best_model, X_train, y_train)
    
    return ModelOutput(
        model_name=best_model_name,
        accuracy=accuracy,
        classification_report=report,
        feature_importances=feature_importances,
        pdp_plots=pdp_plots,
        shap_summary_plot=shap_summary_plot,
        lime_explanation=lime_explanation,
        surrogate_tree=surrogate_tree
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)