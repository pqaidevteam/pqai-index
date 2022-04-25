import unittest
import os
import sys
from pathlib import Path
import numpy as np

TEST_DIR = str(Path(__file__).parent.resolve())
BASE_DIR = str(Path(__file__).parent.parent.resolve())
sys.path.append(BASE_DIR)

test_index_dir = f"{TEST_DIR}/test_indexes"
os.environ["INDEXES_DIRECTORY"] = test_index_dir
os.environ["USE_FAISS_INDEXES"] = "1"
os.environ["USE_ANNOY_INDEXES"] = "1"

from core.indexes import IndexesDirectory
from core.indexes import AnnoyIndexReader, AnnoyIndex, FaissIndexReader, FaissIndex


class TestAnnoyIndexReader(unittest.TestCase):
    def test_read_annoy_index(self):
        """Can read an Annoy index from disk?
        """
        ann_file = f"{test_index_dir}/Y02T.ttl.ann"
        json_file = f"{test_index_dir}/Y02T.ttl.items.json"
        reader = AnnoyIndexReader(768, "angular")
        index = reader.read_from_files(ann_file, json_file)
        self.assertIsInstance(index, AnnoyIndex)


class TestAnnoyIndexClass(unittest.TestCase):
    def setUp(self):
        """Set up an index to use for testing
        """
        ann_file = f"{test_index_dir}/Y02T.ttl.ann"
        json_file = f"{test_index_dir}/Y02T.ttl.items.json"
        reader = AnnoyIndexReader(768, "angular")
        self.index = reader.read_from_files(ann_file, json_file)

    def test_run_query(self):
        """Can it find similar vector for a given query vector?
        """
        qvec = np.ones(768)
        n_results = 10
        results = self.index.search(qvec, n_results)
        self.assertIsInstance(results, list)
        self.assertEqual(n_results, len(results))


class TestFaissIndexReaderClass(unittest.TestCase):
    def test_read_from_files(self):
        """Can read an index from `.faiss` and `.json` files?
        """
        index_file = f"{test_index_dir}/B68G.abs.faiss"
        json_file = f"{test_index_dir}/B68G.abs.items.json"
        r = FaissIndexReader()
        index = r.read_from_files(index_file, json_file)
        self.assertIsInstance(index, FaissIndex)


class TestFaissIndexClass(unittest.TestCase):
    def setUp(self):
        """ The setUp function is called before each test function is run.
        """
        index_file = f"{test_index_dir}/B68G.abs.faiss"
        json_file = f"{test_index_dir}/B68G.abs.items.json"
        r = FaissIndexReader()
        self.index = r.read_from_files(index_file, json_file)

    def test_run_query(self):
        """ Tests the run function of VectorQuery class.
        """
        qvec = np.ones(768)
        n_results = 10
        results = self.index.search(qvec, n_results)
        self.assertIsInstance(results, list)
        self.assertEqual(n_results, len(results))


class TestIndexesDirectory(unittest.TestCase):
    def setUp(self):
        """ The setUp function is called by the test framework at the beginning of each test.  It creates a new instance of your application.
        """
        self.indexes = IndexesDirectory(f"{test_index_dir}")

    def test_get_annoy_indexes(self):
        """Can it discover AnnoyIndexes on disk?
        """
        indexes = self.get_index("Y02T")
        are_index_objects = [isinstance(idx, AnnoyIndex) for idx in indexes]
        self.assertTrue(all(are_index_objects))
        self.assertEqual(3, len(indexes))

    def test_can_get_faiss_indexes(self):
        """Can it find FAISS indexes on the disk?
        """
        indexes = self.get_index("B68G.abs")
        are_index_objects = [isinstance(idx, FaissIndex) for idx in indexes]
        self.assertTrue(all(are_index_objects))
        self.assertEqual(1, len(indexes))

    def test_return_empty_for_inexistent_index(self):
        """Should return empty list if no indexes with given name.
        """
        indexes = self.get_index("Z007")
        self.assertEqual([], indexes)

    def test_available_indexes(self):
        """Discovers indexes on disk
        """
        index_ids = self.indexes.available()
        self.assertIsInstance(index_ids, set)
        self.assertEqual(len(index_ids), 4)

    def get_index(self, index_code):
        """Get an index by its name
        """
        index = self.indexes.get(index_code)
        return index


if __name__ == "__main__":
    unittest.main()
