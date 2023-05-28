"""
Class to store index
"""
import os
from core.indexes import FaissIndexReader, AnnoyIndexReader
import psutil

CHECK_MARK = "\u2713"
USE_FAISS_INDEXES = os.environ["USE_FAISS_INDEXES"]
USE_ANNOY_INDEXES = os.environ["USE_ANNOY_INDEXES"]


class IndexStorage:

    """A collection of indexes read from a directory

    Attributes:
        cache (dict): In-memory store of indexes loaded from the disk
        metric (str): Metric used for similarity, e.g., cosine proximity
    """

    cache = {}
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
        """Get one index by name (either from cache or disk)"""
        if index_id in self.cache:
            return self.cache.get(index_id)
        return self._get_from_disk(index_id)

    def _get_from_disk(self, index_id):
        """Load an index from disk"""
        print(f"Loading vector index: {index_id}")
        index_file = self._get_index_file_path(index_id)
        json_file = f"{self._folder}/{index_id}.metadata.json"
        if index_file.endswith("faiss"):
            reader = FaissIndexReader()
        else:
            reader = AnnoyIndexReader()
        index = reader.read_from_files(index_file, json_file, name=index_id)
        self._cache_index(index_id, index)
        print(
            f"  {CHECK_MARK} RAM usage: {psutil.virtual_memory()._asdict().get('percent')}%"
        )
        return index

    def _get_index_file_path(self, index_id):
        """Get full paths to index file"""
        ann_file = f"{self._folder}/{index_id}.ann"
        faiss_file = f"{self._folder}/{index_id}.faiss"
        return faiss_file if (os.path.exists(faiss_file)) else ann_file

    def _cache_index(self, index_id, index):
        """Store the index in cache"""
        self.cache[index_id] = index

    def available(self):
        """Get a list of index names available in the directory

        Returns:
            list: Index names
        """
        return self._available
