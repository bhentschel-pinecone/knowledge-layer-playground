from typing import List, Dict

import pandas as pd
from datasets import load_dataset

class QRelsDataset:
    def set_documents(self, ids: List[str], texts: List[str]):
        self.doc_ids = ids
        self.doc_texts = texts

    def set_queries(self, ids: List[str], texts: List[str]):
        self.query_ids = ids
        self.query_texts = texts

    def set_qrels(self, qrels_dict: dict):
        self.qrels_dict = qrels_dict

    def get_document_ids_texts(self) -> (List[str], List[str]):
        return self.doc_ids, self.doc_texts

    def get_query_ids_texts(self) -> (List[str], List[str]):
        return self.query_ids, self.query_texts

    def get_qrels(self, set: str):
        return self.qrels_dict[set]

def load_beir_dataset(name: str):
    return_dataset = QRelsDataset()
    for subname in ['corpus', 'queries']:
        hf_dataset = load_dataset(f'BeIR/{name}', subname)
        ids: List[str] = hf_dataset[subname]['_id']
        texts: List[str] = hf_dataset[subname]['text']
        if subname == 'corpus':
            return_dataset.set_documents(ids, texts)
        else:
            return_dataset.set_queries(ids, texts)
    qrels_dataset = load_dataset(f'BeIR/{name}-qrels')
    relevance_dictionary: Dict[str, Dict[str, Dict[str, int]]] = dict() # dictionary: set name (train,validation,test) -> query id -> document id -> score
    for key in qrels_dataset.keys():
        relevance_dictionary[key] = dict()
        qrels = pd.DataFrame(qrels_dataset[key])
        qrels = qrels[qrels['score'] > 0] # keep only positive scores (0 scores don't change ndcg or others)
        for index, row in qrels.iterrows():
            query_id = str(row['query-id'])
            corpus_id = str(row['corpus-id'])
            score = int(row['score'])
            if query_id not in relevance_dictionary[key]:
                relevance_dictionary[key][query_id] = {}
            relevance_dictionary[key][query_id][corpus_id] = score
    return_dataset.set_qrels(relevance_dictionary)
    return return_dataset

class HuggingFaceDatasetLoader:
    @staticmethod
    def get_dataset(name: str):
        if name == "nf":
            dataset = load_beir_dataset("nfcorpus")
        else:
            dataset = None
        return dataset
