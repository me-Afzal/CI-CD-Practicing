"""FastApi App for prediction"""

# import libraries
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI() # Creating the app

# Creating the input data class
class InpData(BaseModel):
    """ Creating the input data class """
    features: list[float]

# Loading the model
model=pickle.load(open('models/model.pkl','rb'))

# Creating the API call for prediction
@app.post("/predict")
def predict(data:InpData):
    """Endpoint for prediction"""
    prediction=model.predict(([data.features]))[0] #Prediction
    prediction=int(prediction) #Converting to int

    return {"prediction":prediction} # Returning the prediction
