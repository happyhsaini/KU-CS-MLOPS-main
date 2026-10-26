from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.post("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_iris_prediction():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_name" in result

    assert result["class_name"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert result["prediction"] in [0, 1, 2]