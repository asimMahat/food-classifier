# import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException  
from prediction import read_imagefile,predict_image,preprocess_image

app = FastAPI()

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not extension:
       raise HTTPException(status_code=400, detail="Please upload image file only")
    image = read_imagefile(await file.read())
    image = preprocess_image(image)
    predictions = predict_image(image)
    return predictions
    