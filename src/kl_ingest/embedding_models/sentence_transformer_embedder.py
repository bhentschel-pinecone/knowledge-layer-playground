from typing import List
import numpy as np
from numpy._typing import NDArray
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from kl_ingest.embedding_models.base import TextEmbeddingModel


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]  # last hidden state layer
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()  # expand so that we can broadcast and cast to float.
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded,
                               1)  # sum all layers where attention_mask is 1 (0 for padded context tokens)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SentenceTransformerEmbedder(TextEmbeddingModel):
    """
    Class for embedding text with the SentenceTransformers library.
    """

    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        if torch.cuda.is_available():
            self.dev = 'cuda:0'
        else:
            self.dev = 'cpu'
        self.device = torch.device(self.dev)
        print(self.device)
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def _embed_strings_batch(self, str_list: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(str_list, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
        inputs = inputs.to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
        output = mean_pooling(model_output, inputs['attention_mask'])
        output = F.normalize(output, p=2, dim=1)
        return output.float().cpu().tolist()

    def _embed_documents_batch(self,
                                documents: List[str]
                                ) -> List[List[float]]:
        return self._embed_strings_batch(documents)

    def _embed_queries_batch(self,
                              queries: List[str]
                              ) -> List[List[float]]:
        return self._embed_strings_batch(queries)
