"""Test the model"""

# Importing the libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Loading the model
model=pickle.load(open('models/model.pkl','rb'))

# Testing the model
def test_model_accuracy():
    """Test the model accuracy"""
    df=pd.read_csv("data/IRIS.csv") # Load dataset

    # Labeling the species
    encoder=LabelEncoder()
    df['species']=encoder.fit_transform(df['species'])

    # Splitting the dataset
    x=df.drop(columns=['species'])
    y=df['species']

    # Train-Test Split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    y_pred=model.predict(x_test) # Predict

    accuracy=accuracy_score(y_test,y_pred) # Accuracy

    assert accuracy>0.8 # Accuracy checking
