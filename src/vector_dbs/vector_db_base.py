from abc import ABC, abstractmethod
from typing import List, Type

from vector_dbs.vector_index_base import VectorIndex


class VectorDatabase(ABC):
    """
    Base class for Vector Databases.
    """

    @abstractmethod
    def get_index(self, name: str) -> Type[VectorIndex]:
        pass

    @abstractmethod
    def create_index(self, name: str) -> Type[VectorIndex]:
        pass

    @abstractmethod
    def delete_index(self, name:str):
        pass

    @abstractmethod
    def get_indexes(self) -> List[str]:
        pass
