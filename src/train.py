""" Training the model and saving it """
# Importing the libraries
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data/IRIS.csv") # Load the dataset

# Labeling the species
encoder=LabelEncoder()
df['species']=encoder.fit_transform(df['species'])

# Splitting the dataset
x=df.drop(columns=['species'])
y=df['species']

# Train-Test Split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Training the model
model=RandomForestClassifier()
model.fit(x_train,y_train)

# Saving the model
pickle.dump(model,open('models/model.pkl','wb'))
