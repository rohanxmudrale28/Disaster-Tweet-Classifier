"""
tests/test_app.py  —  Unit tests for the Disaster Tweet Classifier API
Run with:  pytest tests/ -v
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Make sure app is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, clean_text


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ── clean_text unit tests ─────────────────────────────────────────────────────
class TestCleanText:
    def test_removes_urls(self):
        result = clean_text("Check this out https://example.com")
        assert "http" not in result

    def test_removes_mentions(self):
        result = clean_text("Hello @user how are you")
        assert "@" not in result

    def test_removes_special_chars(self):
        result = clean_text("Fire!!! #disaster2024")
        assert "!" not in result
        assert "#" not in result

    def test_lowercases(self):
        result = clean_text("WILDFIRE ALERT")
        assert result == result.lower()

    def test_strips_extra_spaces(self):
        result = clean_text("fire    near   city")
        assert "  " not in result

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_normal_tweet(self):
        result = clean_text("There is a wildfire near my house!")
        assert isinstance(result, str)
        assert len(result) > 0


# ── /health endpoint ──────────────────────────────────────────────────────────
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'ok'

    def test_health_has_model_loaded_key(self, client):
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'model_loaded' in data


# ── / home endpoint ───────────────────────────────────────────────────────────
class TestHomeEndpoint:
    def test_home_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_home_contains_html(self, client):
        response = client.get('/')
        assert b'Disaster Tweet Classifier' in response.data


# ── /predict endpoint (mocked model) ─────────────────────────────────────────
class TestPredictEndpoint:
    @patch('app.model')
    @patch('app.tokenizer')
    def test_predict_disaster_tweet(self, mock_tokenizer, mock_model, client):
        # Mock tokenizer and model outputs
        mock_tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
        mock_model.predict.return_value = np.array([[0.92]])

        response = client.post('/predict',
                               data=json.dumps({'tweet': 'Wildfire burning near the city!'}),
                               content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['label'] == 1
        assert data['result'] == 'DISASTER'
        assert 0.0 <= data['confidence'] <= 1.0

    @patch('app.model')
    @patch('app.tokenizer')
    def test_predict_not_disaster_tweet(self, mock_tokenizer, mock_model, client):
        mock_tokenizer.texts_to_sequences.return_value = [[4, 5, 6]]
        mock_model.predict.return_value = np.array([[0.12]])

        response = client.post('/predict',
                               data=json.dumps({'tweet': 'I love sunny weather today!'}),
                               content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['label'] == 0
        assert data['result'] == 'NOT DISASTER'

    def test_predict_empty_tweet(self, client):
        response = client.post('/predict',
                               data=json.dumps({'tweet': ''}),
                               content_type='application/json')
        assert response.status_code == 400

    def test_predict_no_body(self, client):
        response = client.post('/predict',
                               data=json.dumps({}),
                               content_type='application/json')
        # Either 400 (empty tweet) or 503 (model not loaded) — both acceptable
        assert response.status_code in [400, 503]

    def test_predict_model_not_loaded(self, client):
        # When model is None (not trained yet)
        import app as app_module
        original_model = app_module.model
        app_module.model = None

        response = client.post('/predict',
                               data=json.dumps({'tweet': 'earthquake disaster'}),
                               content_type='application/json')
        assert response.status_code == 503

        app_module.model = original_model   # restore

    @patch('app.model')
    @patch('app.tokenizer')
    def test_predict_response_has_all_fields(self, mock_tokenizer, mock_model, client):
        mock_tokenizer.texts_to_sequences.return_value = [[1, 2]]
        mock_model.predict.return_value = np.array([[0.75]])

        response = client.post('/predict',
                               data=json.dumps({'tweet': 'flood warning issued'}),
                               content_type='application/json')
        data = json.loads(response.data)
        for field in ['tweet', 'label', 'confidence', 'result']:
            assert field in data, f"Missing field: {field}"
