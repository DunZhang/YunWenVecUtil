import numpy as np
import logging
logger = logging.getLogger(__name__)

class VectorDataBase():
    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.faiss_index = None

    def build_faiss_index(self, index_type="IP"):
        """
        build faiss index
        :param index_type: IP or L2
        :return:
        """
        try:
            import faiss
        except:
            logger.info("no faiss in system")
            return
        logger.info("build faiss index...")
        if index_type == "IP":
            self.faiss_index = faiss.IndexFlatIP(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        elif index_type == "L2":
            self.faiss_index = faiss.IndexFlatL2(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        logger.info("finish building faiss index")
