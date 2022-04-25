"""Index objects and index readers

Attributes:
    CHECK_MARK (str): Unicode symbol for check mark
    USE_ANNOY_INDEXES (1/0): Whether Annoy indexes should be read or ignored
    USE_FAISS_INDEXES (1/0): Whether FAISS indexes should be read or ignored
"""

import numpy as np
import annoy
import json
from abc import ABC, abstractmethod
import os
import faiss
import psutil

CHECK_MARK = "\u2713"
USE_FAISS_INDEXES = os.environ["USE_FAISS_INDEXES"]
USE_ANNOY_INDEXES = os.environ["USE_ANNOY_INDEXES"]


class Index(ABC):

    """An abstract index class
    """

    @abstractmethod
    def search(self, query, n):
        """Return closest matches to a query
        """
        raise NotImplementedError


class VectorIndex(Index):

    """An abstract class for an index that stores vectors
    """

    @abstractmethod
    def search(self, qvec, n):
        """Return mostly similar `n` vectors to query vector `qvec`
        """
        raise NotImplementedError


class AnnoyIndexReader:

    """Loads an Annoy index by reading its `.ann` and `.json` files
    """

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
        """Load index file
        """
        index = annoy.AnnoyIndex(self._dims, self._metric)
        index.load(ann_file)
        return index

    def _get_items_from_json(self, json_file):
        """Read labels
        """
        with open(json_file) as file:
            items = json.load(file)
        return items


class AnnoyIndex(VectorIndex):

    """A wrapper around an AnnoyIndex object
    """

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
        """String representation
        """
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


class FaissIndexReader:

    """Reads Faiss index and associated labels from disk
    """

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
        """Read labels from json file
        """
        with open(json_file) as fp:
            items = json.load(fp)
        return items


class FaissIndex(VectorIndex):

    """A wrapper around an FAISS index
    """

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
        Q = self._preprocess([qvec])
        ds, ns = self._index.search(Q, n)
        items = [self._index2label(i) for i in ns[0]]
        dists = [float(d) for d in ds[0]]
        return list(zip(items, dists))

    # TODO: Move this to indexer
    def add_vectors(self, vectors, labels):
        """Add new vector to an index
        
        Args:
            vectors (list): Vectors to be added
            labels (list): Labels corresponding to the vectors
        """
        assert len(vectors) == len(labels)
        X = self._preprocess(vectors)
        if self._index is None:
            self._init(X)
        self._index.add(X)
        self._labels += labels
        self._save()

    def _init(self, X):
        """Train the index on given vectors
        """
        self._dims = X.shape[1]
        self._labels = []
        self._index = faiss.index_factory(self._dims, "OPQ16_64,HNSW32")
        self._index.train(X)

    def _preprocess(self, vectors):
        """Normalize vectors
        """
        X = np.array(vectors).astype("float32")
        faiss.normalize_L2(X)
        return X

    def _save(self):
        """Save the index to disk
        """
        index_file = f"{self._index_dir}/{self._id}.faiss"
        labels_file = f"{self._index_dir}/{self._id}.labels.json"
        faiss.write_index(self._index, index_file)
        with open(labels_file, "w") as fp:
            json.dump(self._labels, fp)

    @property
    def name(self):
        """Get the index's name
        
        Returns:
            str: Name of the index
        """
        return self._id


class IndexesDirectory:

    """A collection of indexes read from a directory
    
    Attributes:
        cache (dict): In-memory store of indexes loaded from the disk
        dims (int): Dimensionality of the vectors
        metric (str): Metric used for similarity, e.g., cosine proximity
    """

    cache = {}
    dims = 768
    metric = "angular"

    def __init__(self, folder):
        """Initialize
        
        Args:
            folder (path): Directory path where indexes are stored
        """
        self._folder = folder
        self._available = self._discover_indexes()

    def _discover_indexes(self):
        """Scan the directory to find indexes
        
        Returns:
            set: A set of index identifiers (names)
        """
        files = list(os.scandir(self._folder))
        index_files = []
        if USE_FAISS_INDEXES:
            index_files += [f for f in files if f.name.endswith(".faiss")]
        if USE_ANNOY_INDEXES:
            index_files += [f for f in files if f.name.endswith(".ann")]
        index_ids = [".".join(f.name.split(".")[:-1]) for f in index_files]
        return set(index_ids)

    def get(self, index_id):
        """Get an index by name
        
        Args:
            index_id (str): Index's name (or part thereof, i.e. prefix)
        
        Returns:
            list: An array of indexes matching `index_id`
        """
        index_ids = [i for i in self.available() if i.startswith(index_id)]
        indexes = [self._get_one_index(i) for i in index_ids]
        return indexes

    def _get_one_index(self, index_id):
        """Get one index by name (either from cache or disk)
        """
        if index_id in self.cache:
            return self.cache.get(index_id)
        return self._get_from_disk(index_id)

    def _get_from_disk(self, index_id):
        """Load an index from disk
        """
        print(f"Loading vector index: {index_id}")
        index_file = self._get_index_file_path(index_id)
        json_file = f"{self._folder}/{index_id}.items.json"
        if index_file.endswith("faiss"):
            reader = FaissIndexReader()
        else:
            reader = AnnoyIndexReader(self.dims, self.metric)
        index = reader.read_from_files(index_file, json_file, name=index_id)
        self._cache_index(index_id, index)
        print(
            f"  {CHECK_MARK} RAM usage: {psutil.virtual_memory()._asdict().get('percent')}%"
        )
        return index

    def _get_index_file_path(self, index_id):
        """Get full paths to index file
        """
        ann_file = f"{self._folder}/{index_id}.ann"
        faiss_file = f"{self._folder}/{index_id}.faiss"
        return faiss_file if (os.path.exists(faiss_file)) else ann_file

    def _cache_index(self, index_id, index):
        """Store the index in cache
        """
        self.cache[index_id] = index

    def available(self):
        """Get a list of index names available in the directory
        
        Returns:
            list: Index names
        """
        return self._available
