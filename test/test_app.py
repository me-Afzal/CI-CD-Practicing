from fastapi.testclient import TestClient
from app.main import app

client=TestClient(app)

def test_app():
    data={"features":[5.1,3.5,1.4,0.2]}
    response=client.post('/predict',json=data)
    
    assert response.status_code==200
    
    json_data=response.json()
    
    assert 'prediction' in json_data
    
    assert 0<= json_data['prediction']<=2
    