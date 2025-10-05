import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/IRIS.csv")

# Labeling the species

encoder=LabelEncoder()
df['species']=encoder.fit_transform(df['species'])


x=df.drop(columns=['species'])
y=df['species']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

pickle.dump(model,open('models/model.pkl','wb'))