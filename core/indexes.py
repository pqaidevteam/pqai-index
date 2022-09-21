"""Index objects and index readers

Attributes:
    CHECK_MARK (str): Unicode symbol for check mark
    USE_ANNOY_INDEXES (1/0): Whether Annoy indexes should be read or ignored
    USE_FAISS_INDEXES (1/0): Whether FAISS indexes should be read or ignored
"""

import json
from abc import abstractmethod
import numpy as np
import faiss
import annoy


class VectorIndex:

    """Abstract class for indexes that stores vectors"""

    @abstractmethod
    def search(self, qvec: np.ndarray, n: int):
        """Return mostly similar `n` vectors to query vector `qvec`"""
        raise NotImplementedError


class FaissIndex(VectorIndex):

    """A wrapper around an FAISS index"""

    def __init__(self, index, resolver_fn, name=None):
        """Initialize

        Args:
            index (FAISS index object): Index
            resolver_fn (method): Method that returns label for an item id
            name (str, optional): Index's identifier
        """
        self._id = name
        self._index = index
        self._index2label = resolver_fn
        self._labels = None
        self._dims = None

    def search(self, qvec, n):
        """Return `n` most similar items to the given query vector

        Args:
            qvec (list): Query vector
            n (int): No. of items to return

        Returns:
            list: An array of (label, distance) pairs
        """
        Q = np.array([qvec]).astype("float32")
        faiss.normalize_L2(Q)
        ds, ns = self._index.search(Q, n)
        items = [self._index2label(i) for i in ns[0]]
        dists = [float(d) for d in ds[0]]
        return list(zip(items, dists))

    @property
    def name(self):
        """Get the index's name"""
        return self._id


class FaissIndexReader:

    """Reads Faiss index and associated labels from disk"""

    def read_from_files(self, index_file, json_file, name=None):
        """Read index from a `.faiss` and a `.json` file

        Args:
            index_file (path): Vector index file
            json_file (path): JSON file containing vector labels
            name (str, optional): Identifier of index

        Returns:
            FaissIndex: Index object
        """
        index = faiss.read_index(index_file)
        items = self._get_items_from_json(json_file)
        item_resolver = items.__getitem__
        return FaissIndex(index, item_resolver, name)

    def _get_items_from_json(self, json_file):
        """Read labels from json file"""
        with open(json_file) as fp:
            items = json.load(fp)
        return items


class AnnoyIndex(VectorIndex):

    """A wrapper around an AnnoyIndex object"""

    def __init__(self, index: annoy.AnnoyIndex, resolver_fn, name=None):
        """Initialize

        Args:
            index (annoy.AnnoyIndex): Vector index
            resolver_fn (method): Method that returns label for an item id
            name (str, optional): Identifier for the index
        """
        self._index = index
        self._index2item = resolver_fn
        self._name = name
        self._search_depth = 1000

    def search(self, qvec, n):
        """Return `n` most similar items to the given query vector

        Args:
            qvec (list): Query vector
            n (int): No. of items to return

        Returns:
            list: An array of (label, distance) pairs
        """
        d = self._search_depth
        ids, dists = self._index.get_nns_by_vector(qvec, n, d, True)
        items = [self._index2item(i) for i in ids]
        return list(zip(items, dists))

    def set_search_depth(self, d):
        """Set search depth, higher value = more thorough (slower) search

        Args:
            d (int): Search depth
        """
        self._search_depth = d

    def count(self):
        """Return the number of items (vectors) in the index

        Returns:
            int: No. of items in the index
        """
        return self._index.get_n_items()

    def dims(self):
        """Return the dimensionality of vectors present in the index

        Returns:
            int: Dimension count
        """
        v0 = self._index.get_item_vector(0)
        return len(v0)

    def __repr__(self):
        """String representation"""
        idx_type = "AnnoyIndex "
        idx_name = "Unnamed" if self._name is None else self._name
        idx_info = f" [{self.count()} vectors, {self.dims()} dimensions]"
        separator = " "
        return separator.join([idx_type, idx_name, idx_info])

    @property
    def name(self):
        """Return name of the index

        Returns:
            str: Index's name
        """
        return self._name


class AnnoyIndexReader:

    """Loads an Annoy index by reading its `.ann` and `.json` files"""

    def __init__(self, dims: int, metric: str):
        """Initialize

        Args:
            dims: Dimension of stored vectors, e.g., 32
            metric: Distance metric, e.g., 'cosine'
        """
        self._dims = dims
        self._metric = metric

    def read_from_files(self, ann_file: str, json_file: str, name=None):
        """
        Args:
            ann_file (path): Annoy index file
            json_file (path): JSON file containing labels for index vectors
            name (str, optional): Index's name

        Returns:
            annoy.AnnoyIndex: Index
        """
        index = self._read_ann(ann_file)
        items = self._get_items_from_json(json_file)
        item_resolver = items.__getitem__
        return AnnoyIndex(index, item_resolver, name)

    def _read_ann(self, ann_file):
        """Load index file"""
        index = annoy.AnnoyIndex(self._dims, self._metric)
        index.load(ann_file)
        return index

    def _get_items_from_json(self, json_file):
        """Read labels"""
        with open(json_file) as file:
            items = json.load(file)
        return items
