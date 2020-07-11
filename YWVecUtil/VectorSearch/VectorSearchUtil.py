import scipy.spatial
from sklearn.preprocessing import normalize
import numpy as np
from ..SentenceEncoder import ISentenceEncoder
from ..VectorDataBase import VectorDataBase
import logging
logger = logging.getLogger(__name__)

def find_topk_by_vecs(source_vecs, vec_db: VectorDataBase, topk, metric="cosine", use_faiss="auto"):
    """
    find topk vecotr
    :param source_vecs:
    :param vec_db: target vector
    :param topk: topk result
    :param metric: see scipy.spatial.cdist
    :param use_faiss: bool,str, True or False or "Auto"
    :return: res_index and res_distance
        topk's index and instance
    """
    if isinstance(use_faiss, str) and use_faiss.lower() == "auto":
        use_faiss = True
        try:
            import faiss
        except:
            logger.info("faiss is not availble")
            use_faiss = False
    res_distance, res_index = None, None
    if use_faiss:
        if metric == "cosine":
            source_vecs = normalize(source_vecs, axis=1)
            if vec_db.faiss_index is None:
                vec_db.build_faiss_index("IP")
            if vec_db.faiss_index is not None:
                res_distance, res_index = vec_db.faiss_index.search(source_vecs, topk)
        elif metric == "euclidean":
            if vec_db.faiss_index is None:
                vec_db.build_faiss_index("L2")
            if vec_db.faiss_index is not None:
                res_distance, res_index = vec_db.faiss_index.search(source_vecs, topk)
    if res_index is None:
        logger.info("use scipy to search")
        sims = scipy.spatial.distance.cdist(source_vecs, vec_db.vector, metric)
        res_index = np.argsort(sims, axis=1)[:, 0: topk]
        res_distance = np.ones(shape=res_index.shape, dtype=np.float)
        for i in range(res_index.shape[0]):
            for j in range(res_index.shape[1]):
                if metric == "cosine":
                    res_distance[i, j] = 1 - sims[i, res_index[i, j]]
                else:
                    res_distance[i, j] = sims[i, res_index[i, j]]
    return res_index, res_distance


def find_topk_by_sens(sen_encoder: ISentenceEncoder, source_sens, target_sens, topk,
                      metric="cosine", use_faiss="auto"):
    """
    :param sen_encoder: ISentenceEncoder
    :param source_sens: List-like [sen1,sen2,....]
    :param target_sens: List-like [sen1,sen2,....]
    :param topk:
    :param metric:
    :param use_faiss: True or False or auto
    :return: [[sen,topk sens,topk sens's similarity],..]
    """
    source_vecs = sen_encoder.get_sens_vec(source_sens)
    vec_db = VectorDataBase(sen_encoder.get_sens_vec(target_sens))
    res_index, res_distance = find_topk_by_vecs(source_vecs, vec_db, topk, metric, use_faiss)
    ### format data
    data = []
    for i in range(res_index.shape[0]):
        t = [source_sens[i], [], []]
        for j in range(res_index.shape[1]):
            t[1].append(target_sens[res_index[i, j]])
            t[2].append(res_distance[i, j])
        data.append(t)
    return data
