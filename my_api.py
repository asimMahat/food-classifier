# import uvicorn
from fastapi import FastAPI 
from fastapi import UploadFile, File  
from prediction import read_imagefile,predict,preprocess

app = FastAPI()

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print("The image file must be in (.png, .jpg or .jpeg) format")
    image = read_imagefile(await file.read())
    image = preprocess(image)
    predictions = predict(image)
    # print(predictions)
    return predictions
    