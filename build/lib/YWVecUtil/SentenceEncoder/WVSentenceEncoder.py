import gensim
import numpy as np
from .ISentenceEncoder import ISentenceEncoder
from typing import Iterable
import logging
logger = logging.getLogger(__name__)
class WVSentenceEncoder(ISentenceEncoder):
    """
    weighted average vector
    """

    def __init__(self, wv_model_or_path, binary: bool = False, limit: int = None):
        """
        Parameters
        ----------
        wv_model_or_path : Word2VecKeyedVectors or str
        binary : bool, optional
            If True, indicates whether the data is in binary word2vec format.
        limit : int, optional
            Sets a maximum number of word-vectors to read from the file. The default,
            None, means read all.

        """
        if isinstance(wv_model_or_path, str):
            self.wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_model_or_path, binary=binary,
                                                                            limit=limit)
        else:
            self.wv_model = wv_model_or_path
        self.num_dim = self.wv_model.vector_size
        logger.info("vec dim:{}".format(self.num_dim))
    def get_sens_vec(self, sens_words: Iterable[Iterable[str]],
                     words_weight: Iterable[Iterable[float]] = None) -> np.ndarray:
        """
        Parameters
        ----------
        sens_words :  Iterable[Iterable[str]]
            [[word1,word2,...], [word1,word2,word3,...],...]
        words_weight : Iterable[Iterable[float]]
            words' weight

        Returns
        -------
        np.ndarray, shape: num_sen*num_dim
        """
        sens_vec = []
        if words_weight is None:
            words_weight = [[1] * len(i) for i in sens_words]
        for sen, ww in zip(sens_words, words_weight):
            sen_vec = []
            for word, weight in zip(sen, ww):
                if word in self.wv_model.vocab:
                    sen_vec.append(self.wv_model[word] * weight)
            if len(sen_vec) > 0:
                sen_vec = np.array(sen_vec)
                sen_vec = np.mean(sen_vec, axis=0, keepdims=True)
            else:
                sen_vec = np.random.rand(1, self.num_dim)
            sens_vec.append(sen_vec)
        return np.vstack(sens_vec).astype("float32")
