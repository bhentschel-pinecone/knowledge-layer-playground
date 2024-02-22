from typing import List, Tuple

from evaluators import ndcg_evaluation

import evaluators.ndcg_evaluation
from kl_ingest.datasets import HuggingFaceDatasetLoader
from kl_ingest.embedding_models.sentence_transformer_embedder import SentenceTransformerEmbedder
from vector_dbs.local_chroma import ChromaDatabase

def batch_generator(df, batch_size):
    """Yield successive n-sized chunks from df."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

# data_loader = DataLoader()
# df_files = data_loader.parquet_folder_to_pandas("/home/brianhentschel/knowledge-layer-playground/data")
# print(df_files[0].head())
# embedding_model = SentenceTransformerEmbedder()
# for df in df_files:
#     for batch in tqdm.tqdm(batch_generator(df, 2048)):
#         ids = df['docid'].tolist()
#         texts = df['text'].tolist()
#         embeddings: List[List[float]] = embedding_model.encode_documents(texts)
#         metadatas = [dict()] * len(embeddings)
#         test_index.upsert_many(ids, embeddings, metadatas)

if __name__ == "__main__":
    # building chroma database
    print("Connecting to chroma database")
    chroma_client = ChromaDatabase('./chroma_data/')
    print("Creating chroma index.")
    if chroma_client.get_index('test') != None:
        print("Deleting old index")
        chroma_client.delete_index('test')
    test_index = chroma_client.create_index('test', 'cosine')
    # load data from huggingface
    data = HuggingFaceDatasetLoader.get_dataset("nf")
    # create embedding model
    embedding_model = SentenceTransformerEmbedder()
    # iterate through data items and insert them to index.
    ids, texts = data.get_document_ids_texts()
    print(f'Embedding {len(ids)} ids')
    embeddings = embedding_model.embed_documents(texts)
    for (doc_id, embedding) in zip(ids, embeddings):
        test_index.upsert(doc_id, embedding, {})
    print("done inserting data")
    # iterate through queries and issue vector db queries.
    query_ids, query_texts = data.get_query_ids_texts()
    print(f'The number of queries is {len(query_ids)}')
    qrels_dict = data.get_qrels('validation')
    print(f'Number of keys in qrels dict is {len(qrels_dict.keys())}')
    filtered_query_ids, filtered_texts = ndcg_evaluation.filter_query_set(query_ids, query_texts, qrels_dict)
    print(f'After filtering: {len(filtered_query_ids)}')
    query_embeddings = embedding_model.embed_queries(filtered_texts)
    metadata_filters = [{}] * len(filtered_query_ids)
    query_results: List[Tuple[List[str], List[float]]] = test_index.query_many(10, query_embeddings, metadata_filters)
    pytrec_dictionary = ndcg_evaluation.build_pytrec_dictionary(filtered_query_ids, query_results)
    # evaluate ndcg on returned results.
    ndcg = ndcg_evaluation.pytrec_ndcg(qrels_dict, pytrec_dictionary, [10])
    print(f'The ndcg is {ndcg}')