from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class InpData(BaseModel):
    features: list[float]
    
model=pickle.load(open('models/model.pkl','rb'))

@app.post("/predict")
def predict(data:InpData):
    prediction=model.predict(([data.features]))[0]
    prediction=int(prediction)
    
    return {"prediction":prediction}    