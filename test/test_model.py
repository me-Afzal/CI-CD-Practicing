import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

model=pickle.load(open('models/model.pkl','rb'))

def test_model_accuracy():
    df=pd.read_csv("data/IRIS.csv")
    # Labeling the species

    encoder=LabelEncoder()
    df['species']=encoder.fit_transform(df['species'])
    
    x=df.drop(['species'])
    y=df['species']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    y_pred=model.predict(x_test)

    accuracy=accuracy_score(y_test,y_pred)
    
    assert accuracy>0.8

