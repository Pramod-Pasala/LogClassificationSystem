from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import pandas as pd
import os
from classify import classify_log, classify_csv

app = FastAPI()

@app.post("/classify-log/")
async def classify_log_endpoint(source: str = Form(...), log_message: str = Form(...)):
    """
    Endpoint to classify a single log message.
    """
    label = classify_log(source, log_message)
    return {"source": source, "log_message": log_message, "label": label}

@app.post("/classify-csv/")
async def classify_csv_endpoint(file: UploadFile):
    """
    Endpoint to classify logs in a CSV file.
    """
    input_file_path = f"temp_{file.filename}"
    output_file_path = f"classified_{file.filename}"

    # Save the uploaded file temporarily
    with open(input_file_path, "wb") as f:
        f.write(await file.read())

    # Classify the logs in the CSV file
    classify_csv(input_file_path, output_file_path)

    # Remove the temporary input file
    os.remove(input_file_path)

    # Return the classified CSV file
    return FileResponse(output_file_path, media_type="text/csv", filename=output_file_path)