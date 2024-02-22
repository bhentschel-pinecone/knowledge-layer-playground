from typing import List, Type, Dict, Tuple, Optional

from generic_utils import batch_iterator
from vector_dbs.vector_db_base import VectorDatabase
from vector_dbs.vector_index_base import VectorIndex

import chromadb


class ChromaIndex(VectorIndex):

    def __init__(self, collection: chromadb.Collection, metric = 'euclidean'):
        self.index = collection
        self.batch_size = 5000
        self.metric = metric

    # @staticmethod
    # # add back -> Self when Python 3.11
    # def create_index(name: str, client: chromadb.api.BaseAPI, metric: str = "cosine"):
    #     if metric == "euclidean":
    #         index = client.create_collection(name=name)
    #     elif metric == "cosine":
    #         index = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    #     else:
    #         raise ValueError("metric must be either cosine or euclidean")
    #     return ChromaIndex(index)

    # @staticmethod
    # # add back -> Self when Python 3.11
    # def load_index(name: str, client: chromadb.api.BaseAPI):
    #     index = client.get_collection(name)
    #     return ChromaIndex(index)

    def upsert(self, id: str, vec: List[float], metadata: Dict):
        self.index.upsert(embeddings=[vec],
                          ids=[id])

    def upsert_many(self, ids: List[str], vecs: List[List[float]], metadatas: List[Dict]):
        for batch in batch_iterator([ids, vecs, metadatas], self.batch_size):
            self.index.upsert(
                ids=batch[0],
                embeddings=batch[1])

    def query(self, k: int, query_vec: List[float], metadata_filter: Dict) -> (List[str], List[float]):
        return_dict = self.index.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=metadata_filter
        )
        # only one query, so take the first item.
        return_str = return_dict['ids'][0]
        return_floats = return_dict['distances'][0]
        if self.metric == 'cosine':
            return_floats = [-x + 1.0 for x in return_floats]
        return return_str, return_floats

    def query_many(self, k: int, query_vecs: List[List[float]], metadata_filters: List[Dict]) \
            -> List[Tuple[List[str], List[float]]]:
        results = []
        for (query_vec, metadata_filter) in zip(query_vecs, metadata_filters):
            result = self.query(k, query_vec, metadata_filter)
            results.append(result)
        print("PRINTING FIRST RESULT")
        print(results[0])
        return results

    def get(self, id: str):
        self.index.get(ids=[id])


class ChromaDatabase(VectorDatabase):
    """
    Base class for Vector Databases.
    """

    def __init__(self, save_path: str):
        self.client = chromadb.PersistentClient(path=save_path)
        self.indexes: Dict[str, ChromaIndex] = dict()
        for collection in self.client.list_collections():
            self.indexes[collection.name] = ChromaIndex(collection)

    def get_index(self, name: str) -> Optional[ChromaIndex]:
        return self.indexes.get(name)

    def create_index(self, name: str, metric: str = "cosine") -> ChromaIndex:
        if metric == "euclidean":
            print("creating euclidean index")
            index = self.client.create_collection(name=name)
        elif metric == "cosine":
            print("creating cosine index")
            index = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        else:
            raise ValueError("metric must be either cosine or euclidean")
        index = ChromaIndex(index, metric)
        self.indexes[name] = index
        return index

    def delete_index(self, name: str):
        self.client.delete_collection(name)
        # might use the missing key later in bettter software development.
        self.indexes.pop(name, 'No Key found')

    def get_indexes(self) -> List[str]:
        return list(self.indexes.keys())
