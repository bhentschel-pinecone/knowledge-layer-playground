from abc import ABC, abstractmethod
from typing import List, Optional

from numpy._typing import NDArray
import numpy as np


class TextEmbeddingModel(ABC):
    """
    Base class for TextEmbedding Models. Encodes text and queries to vector representations.
    """

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    @abstractmethod
    def _encode_documents_batch(self,
                                documents: List[str]
                                ) -> List[NDArray[np.float32]]:
        pass

    @staticmethod
    def _batch_iterator(data: list, batch_size):
        return (data[pos:pos + batch_size] for pos in range(0, len(data), batch_size))

    @abstractmethod
    def _encode_queries_batch(self, queries: List[str]) -> List[NDArray[np.float32]]:
        pass

    def encode_documents(self, documents: List[str]) -> List[NDArray[np.float32]]:
        encoded_docs = []
        for batch in self._batch_iterator(documents, self.batch_size):
            try:
                encoded_docs.extend(self._encode_documents_batch(batch))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enconde documents using {self.__class__.__name__}. "
                    f"Error: {self._format_error(e)}"
                ) from e

        return encoded_docs  # TODO: consider yielding a generator

    def encode_queries(self, queries: List[str]) -> List[NDArray[np.float32]]:
        """

        Encode queries in batches. Will iterate over batch of queries and encode them using the _encode_queries_batch method.

        Args:
            queries: A list of Query to encode.

        Returns:
            encoded queries: A list of KBQuery.
        """  # noqa: E501

        kb_queries = []
        for batch in self._batch_iterator(queries, self.batch_size):
            try:
                kb_queries.extend(self._encode_queries_batch(batch))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to enconde queries using {self.__class__.__name__}. "
                    f"Error: {self._format_error(e)}"
                ) from e

        return kb_queries

    def _format_error(self, err):
        return f"{err}"

    @property
    def dimension(self) -> Optional[int]:
        """
        Returns:
            The dimension of the dense vectors produced by the encoder, if applicable.
        """  # noqa: E501
        return None

