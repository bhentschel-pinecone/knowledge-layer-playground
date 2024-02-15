import numpy as np
from typing import List, Dict, Tuple
import pytrec_eval
import logging

class Evaluators:
    # code taken (with modifications) from BEIR
    @staticmethod
    def pytrec_ndcg(qrels: Dict[str, Dict[str, int]],
                 results: Dict[str, Dict[str, float]],
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Dict[int, float]:

        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}

        for k in k_values:
            ndcg[k] = 0.0

        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[k] += scores[query_id]["ndcg_cut_" + str(k)]

        for k in k_values:
            ndcg[k] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)

        for eval in [ndcg]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("ncdg@{}: {:.4f}".format(k, eval[k]))

        return ndcg

    @staticmethod
    def manual_calculate_ndcg(relevance_dictionary: dict[str, dict[str, int]], NDCG_cutoff: int, int_labels: np.array, query_ids: list[str], data_ids: list[str]):
        max_gains = np.zeros(len(query_ids), dtype=np.float32)
        for index, query in enumerate(query_ids):
            sorted_relevance_list = sorted(list(relevance_dictionary[query].items()), key=lambda x: x[1], reverse=True)
            sorted_relevances = np.array([x[1] for x in sorted_relevance_list])
            sorted_relevances.resize(NDCG_cutoff)
            discounted_gain = np.sum(sorted_relevances / np.log2(np.arange(2, sorted_relevances.shape[0] + 2)))
            max_gains[index] = discounted_gain
        ndcgs = []
        for query, returned_ids, max_gain in zip(query_ids, int_labels, max_gains):
            query_relevances_dict = relevance_dictionary[query]
            returned_ids = returned_ids.tolist()
            data_labels = [data_ids[returned_id] for returned_id in returned_ids]
            data_relevances = np.array([query_relevances_dict.get(x, 0) for x in data_labels])
            data_relevances.resize(NDCG_cutoff)
            total_discounted_gain = np.sum(data_relevances / np.log2(np.arange(2, data_relevances.shape[0] + 2)))
            if max_gain > 0:
                ndcg = total_discounted_gain / max_gain
            else:
                ndcg = 0
            # this shouldn't happen, even taking into account floating point precision
            # 1.0 will see print statements for ndcg 1.00002 and the like
            if ndcg > 1.01:
                print("query: {}, ndcg is {}".format(query, ndcg))
                print(query_relevances_dict)
                print(data_labels)
                print(data_relevances)
            ndcgs.append(ndcg)
        avg_ncdg = np.mean(ndcgs)
        print(f'avg_ncdg: {avg_ncdg}')
        return avg_ncdg