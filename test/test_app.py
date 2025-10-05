""" Test the app"""

# import libraries
from fastapi.testclient import TestClient
from app.main import app

# Setting client
client=TestClient(app)

# Testing the app
def test_app():
    """ Test the app"""
    data={"features":[5.1,3.5,1.4,0.2]} # Dummy data
    response=client.post('/predict',json=data) # API call

    assert response.status_code==200 # Status code checking

    json_data=response.json() # JSON data

    assert 'prediction' in json_data # Key checking

    assert 0<= json_data['prediction']<=2 # Value checking
    