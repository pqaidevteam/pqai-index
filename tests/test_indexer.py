"""
    Tests for indexers
"""

import unittest
import os
from pathlib import Path
import sys
import numpy as np

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
sys.path.append(BASE_DIR)

from core.indexer import FaissIndexCreator

class TestFaissIndexCreator(unittest.TestCase):

    def setUp(self):
        n_vectors = 20000
        n_dims = 128
        shape = (n_vectors, n_dims)
        self.vectors = np.random.normal(size=shape).astype("float32")
        self.labels = [str(i) for i in range(n_vectors)]
        self.index_name = "test_index"
        self.save_dir = TEST_DIR
        self.cleanup()

    @unittest.skip("temporary")
    def test__can_create_index(self):
        index_file = f"{self.save_dir}/{self.index_name}.faiss"
        config_file = f"{self.save_dir}/{self.index_name}.config.json"

        self.assertFalse(os.path.exists(index_file))
        self.assertFalse(os.path.exists(config_file))

        options = {
            "normalize": True,
            "factory_string": "OPQ16_64,HNSW32"
        }
        creator = FaissIndexCreator(**options)
        index = creator.create(
            name=self.index_name,
            vectors=self.vectors,
            labels=self.labels,
            n_train=None,
            save_dir=TEST_DIR
        )
        self.assertTrue(os.path.exists(index_file))
        self.assertTrue(os.path.exists(config_file))
        self.cleanup()

    def test__error_if_labels_vectors_mismatch(self):
        options = {
            "normalize": True,
            "factory_string": "OPQ16_64,HNSW32"
        }
        creator = FaissIndexCreator(**options)
        attempt = lambda: creator.create(
            name=self.index_name,
            vectors=self.vectors,
            labels=self.labels[:-1],
            n_train=None,
            save_dir=TEST_DIR
        )
        self.assertRaises(Exception, attempt)

    def cleanup(self):
        index_file = f"{self.save_dir}/{self.index_name}.faiss"
        config_file = f"{self.save_dir}/{self.index_name}.config.json"
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(config_file):
            os.remove(config_file)

if __name__ == "__main__":
    unittest.main()
