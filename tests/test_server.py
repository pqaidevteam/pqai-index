"""Test for service API
"""
import unittest
import json
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from fastapi.testclient import TestClient

BASE_PATH = Path(__file__).parent.parent.resolve()
ENV_PATH = BASE_PATH / ".env"

load_dotenv(ENV_PATH.as_posix())
sys.path.append(BASE_PATH.as_posix())

from main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test__can_search(self):
        params = {
            "mode": "vector",
            "query": json.dumps(list(np.random.random(384))),
            "n": 5,
        }
        r = self.client.get("search", params=params)
        self.assertEqual(200, r.status_code)
        self.assertIsInstance(r.json(), dict)

        results = r.json().get("results")
        self.assertIsInstance(results, list)
        self.assertEqual(5, len(results))

        # Make sure each result is a [label, score)] pair
        self.assertTrue(all([isinstance(r, list) for r in results]))
        self.assertTrue(all([len(r) == 2 for r in results]))
        self.assertTrue(all([isinstance(label, str) for label, _ in results]))
        self.assertTrue(all([isinstance(score, float) for _, score in results]))

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


if __name__ == "__main__":
    unittest.main()
