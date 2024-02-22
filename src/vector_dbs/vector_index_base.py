from abc import ABC, abstractmethod
from typing import List, Dict


class VectorIndex(ABC):
    """
    Base class for Vector Databases.
    """

    @abstractmethod
    def upsert(self, id: str, vec: List[float], metadata: Dict):
        pass

    def upsert_many(self, ids: List[str], vecs: List[List[float]], metadatas: List[Dict]):
        for (id, vec, metadata) in zip(ids, vecs, metadatas):
            self.upsert((id, vec, metadata))


    @abstractmethod
    def query(self, k: int, query_vec: List[float], metadata_filter: Dict) -> List[str]:
        """
        Queries the index and returns a list of ids (currently). To think about the interface.

        :param k: Number of neighbors to get.
        :param query_vec: Embedding to look for nearest neighbors of.
        :param metadata_filter: A dictionary of {metadata_field: {operator: value}}.
        The operators are $eq, $ne, $gt, $gte, $lt, $lte. $eq,$ne allowed on string, int, float. The comparisons only on int, float.
        To figure out: in, notin (supported by pinecone so seems like we should be able to do it).
        """
        pass

    def query_many(self, k: int, query_vecs: List[List[float]], metadata_filters: List[Dict]) -> List[List[str]]:
        results = []
        for (query_vec, metadata_filter) in zip(query_vecs, metadata_filters):
            results.append(self.query(k, query_vec, metadata_filter))
        return results

    @abstractmethod
    def get(self, id: str):
        pass
