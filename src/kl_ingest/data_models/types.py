from dataclasses import dataclass
from typing import List, Any


@dataclass
class Document:
    """ A document contains a document id as well as the text inside the document.
    """
    id: str
    text: str

    @staticmethod
    def doc_list_to_split_lists(docs) -> (List[str], List[str]):
        id_list = []
        text_list = []
        for doc in docs:
            id_list.append(doc.id)
            text_list.append(doc.text)
        return (id_list, text_list)

    @staticmethod
    def batch_iterate_split(docs, batch_size):
        pass

@dataclass(frozen=True)
class Embedding:
    """An embedded unit of a file.
    """
    id: str
    doc_id: str
    text: str
    # begin_offset: int
    # end_offset: int
    vector: List[float]
    metadata: dict[str, Any]

    def get_document_id(self):
        return self.doc_id

    def get_id(self):
        return self.id

    # def get_offsets(self):
    #     return (self.begin_offset, self.end_offset)

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata