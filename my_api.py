from fastapi import FastAPI 
from fastapi import UploadFile, File  
import uvicorn
from prediction import read_imagefile,predict,preprocess


app = FastAPI()


# @app.get("/index")
# # def index(name:str):
# #     return f"Hello {name}! " 

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)):
    # Read the file uploaded by the user
    image = read_imagefile(await file.read())

    # Do input image preprocessing 
    image = preprocess(image)

    # make predictions 
    predictions = predict(image)
    print(predictions)
    return predictions


# @app.get("/index")
# def index(name:str):
#     return f"Hello {name}! " 

# @app.post("/predict/image")
# async def predict_api(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     if not extension:
#         return "Image must be jpg or png format!"
#     image = read_imagefile(await file.read())
#     prediction = predict(image)
#     return prediction

if __name__=="__main__":
    uvicorn.run(app, port=8080, host ='0.0.0.0')

