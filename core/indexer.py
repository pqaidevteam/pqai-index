"""
Classes used for creating indexes
"""

import json
import numpy as np
import faiss

class FaissIndexCreator:

    def __init__(self, factory_string, normalize=True):
        self.factory_string = factory_string,
        self.normalize = normalize

    def create(self, name: str, vectors: np.ndarray, labels: list, n_train: int, save_dir: str):
        assert isinstance(vectors, np.ndarray)
        assert vectors.dtype == np.float32
        assert len(vectors.shape) == 2
        assert len(vectors) == len(labels)

        n_vectors, n_dims = vectors.shape
        index = faiss.index_factory(n_dims, "OPQ16_64,HNSW32")

        if self.normalize:
            faiss.normalize_L2(vectors)
        
        index.train(vectors[:n_train])
        index.add(vectors)
        config = {
            "name": name,
            "factory_string": self.factory_string,
            "normalized": self.normalize,
            "dims": n_dims,
            "item_count": n_vectors,
            "labels": labels
        }
        self._save(index, config, save_dir)

    def _save(self, index, config, save_dir):
        """Save index to disk"""
        name = config["name"]
        faiss.write_index(index, f"{save_dir}/{name}.faiss")
        with open(f"{save_dir}/{name}.config.json", "w") as fp:
            json.dump(config, fp)


class AnnoyIndexCreator:

    def __init__(self):
        raise NotImplementedError

    def create(self, name: str, vectors: np.ndarray, labels: list, n_train: int, save_dir: str):
        raise NotImplementedError
