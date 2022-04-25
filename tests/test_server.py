"""Test for service API

Attributes:
    dotenv_file (str): Absolute path to .env file (used for reading port no.)
    HOST (str): IP address of the host where service is running
    PORT (str): Port no. on which the server is listening
    PROTOCOL (str): `http` or `https`
"""
import unittest
import os
import json
import re
import socket
from pathlib import Path
import numpy as np
import dotenv
import requests

dotenv_file = str((Path(__file__).parent.parent / ".env").resolve())
dotenv.load_dotenv(dotenv_file)

PROTOCOL = "http"
HOST = "localhost"
PORT = os.environ["PORT"]
assert re.match(r"^\d+$", PORT)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_not_running = sock.connect_ex((HOST, int(PORT))) != 0
if server_not_running:
    print("Server is not running. API tests will be skipped.")


@unittest.skipIf(server_not_running, "Works only when true")
class TestAPI(unittest.TestCase):

    """Check if all API routes are working as expected
    """

    def test_can_search(self):
        """Check if a valid response is returned for a legit request
        """
        params = {
            "mode": "vector",
            "query": json.dumps(list(np.random.random(768))),
            "n": 5,
        }
        response = self.call_route("/search", params)
        self.assertEqual(200, response.status_code)
        self.assertIsInstance(response.json(), dict)

        results = response.json().get("results")
        self.assertIsInstance(results, list)
        self.assertEqual(5, len(results))

        # Make sure each result is a [label, score)] pair
        self.assertTrue(all([isinstance(r, list) for r in results]))
        self.assertTrue(all([len(r) == 2 for r in results]))
        self.assertTrue(all([isinstance(label, str) for label, _ in results]))
        self.assertTrue(all([isinstance(score, float) for _, score in results]))

    def test_invalid_mode_in_request(self):
        """Make sure HTTP 400 is returned when search mode is invalid
        """
        params = {
            "mode": "invalid_mode",
            "query": json.dumps(list(np.random.random(768))),
            "n": 5,
        }
        response = self.call_route("/search", params)
        self.assertEqual(400, response.status_code)
        self.assertEqual("Invalid search mode", response.json().get("detail"))

    def test_invalid_json_encoded_vector_in_request(self):
        """Make sure HTTP 400 is returned when JSON encoded query vector is not
        parsable
        """
        params = {"mode": "vector", "query": "invalid_json", "n": 5}
        response = self.call_route("/search", params)
        self.assertEqual(400, response.status_code)
        self.assertEqual("Invalid JSON query", response.json().get("detail"))

    def call_route(self, route, params):
        """Make a request to given route with given parameters
        
        Args:
            route (str): Route, e.g. '/search'
            params (dict): Query string parameters
        
        Returns:
            requests.models.Response: Response against HTTP request
        """
        route = route.lstrip("/")
        url = f"{PROTOCOL}://{HOST}:{PORT}/{route}"
        response = requests.get(url, params)
        return response


if __name__ == "__main__":
    unittest.main()
