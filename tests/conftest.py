import pytest
from fastapi.testclient import TestClient
from src.app import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
