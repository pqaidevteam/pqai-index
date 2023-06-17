"""Test for service API
"""
import unittest
import json
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError

BASE_PATH = Path(__file__).parent.parent.resolve()
ENV_PATH = BASE_PATH / ".env"

load_dotenv(ENV_PATH.as_posix())
sys.path.append(BASE_PATH.as_posix())

from main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.results_schema = {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2
            },
        }

    def test__can_search_without_specifying_index(self):
        params = {
            "mode": "vector",
            "query": json.dumps(list(np.random.random(384))),
            "n": 5,
        }
        r = self.client.get("search", params=params)
        self.assertEqual(200, r.status_code)
        self.assertIsInstance(r.json(), dict)

        results = r.json().get("results")
        self.followsJsonSchema(results, self.results_schema)

    def test__can_search_in_specific_index(self):
        params = {
            "mode": "vector",
            "query": json.dumps(list(np.random.random(384))),
            "n": 5,
            "index": "drones"
        }
        r = self.client.get("search", params=params)
        self.assertEqual(200, r.status_code)
        self.assertIsInstance(r.json(), dict)

        results = r.json().get("results")
        self.followsJsonSchema(results, self.results_schema)

    def test__invalid_mode_in_request(self):
        """Make sure HTTP 400 is returned when search mode is invalid"""
        params = {
            "mode": "invalid_mode",
            "query": json.dumps(list(np.random.random(768))),
            "n": 5,
        }
        r = self.client.get("/search", params=params)
        self.assertEqual(400, r.status_code)
        self.assertEqual("Invalid search mode", r.json().get("detail"))

    def test__invalid_json_encoded_vector_in_request(self):
        """Make sure HTTP 400 is returned when JSON encoded query vector is not
        parsable
        """
        params = {"mode": "vector", "query": "invalid_json", "n": 5}
        r = self.client.get("/search", params=params)
        self.assertEqual(400, r.status_code)
        self.assertEqual("Invalid JSON query", r.json().get("detail"))
    
    def followsJsonSchema(self, data, schema):
        try:
            validate(data, schema)
        except ValidationError as e:
            self.fail(f"Invalid results schema: {e}")


if __name__ == "__main__":
    unittest.main()
