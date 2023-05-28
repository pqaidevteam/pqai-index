"""
Classes used for creating indexes
"""

import json
import numpy as np
import faiss


class FaissIndexCreator:

    """Puts a list of vectors in an FAISS index and saves it to disk"""

    def __init__(self, factory_string: str, normalize: bool = True):
        """Initialise

        Args:
            factory_string (str): Indexing configuration (see FAISS docs)
            normalize (bool, optional): Convert to unit vectors before indexing
        """
        self._factory_string = factory_string
        self._normalize = normalize

    def create(
        self, name: str, vectors: np.ndarray, labels: list, n_train: int, save_dir: str
    ):
        """Create an index with given vectors and save to disk

        Args:
            name (str): Index's name
            vectors (np.ndarray): 2D matrix, rows are vectors
            labels (list): Textual labels of vectors in the same order
            n_train (int, optional): Number of vectors to be used for training
                the indexer. If it is `None` all of them are used for training
            save_dir (str): Absolute path to the directory where the index
                files will be saved
        """
        assert isinstance(vectors, np.ndarray)
        assert vectors.dtype == np.float32
        assert len(vectors.shape) == 2
        assert len(vectors) == len(labels)

        n_vectors, n_dims = vectors.shape
        index = faiss.index_factory(n_dims, self._factory_string)

        if self._normalize:
            faiss.normalize_L2(vectors)

        index.train(vectors[:n_train])
        index.add(vectors)
        config = {
            "name": name,
            "factory_string": self._factory_string,
            "normalized": self.normalize,
            "dims": n_dims,
            "item_count": n_vectors,
            "labels": labels,
        }
        self._save(index, config, save_dir)

    def _save(self, index, config: dict, save_dir: str):
        """Save index to disk"""
        name = config["name"]
        faiss.write_index(index, f"{save_dir}/{name}.faiss")
        with open(f"{save_dir}/{name}.metadata.json", "w") as fp:
            json.dump(config, fp)


class AnnoyIndexCreator:

    """Puts an array of vectors in an Annoy index and saves it to disk"""

    def __init__(self):
        """Initialise"""
        raise NotImplementedError

    def create(
        self, name: str, vectors: np.ndarray, labels: list, n_train: int, save_dir: str
    ):
        """Create an index with given vectors and save to disk"""
        raise NotImplementedError
