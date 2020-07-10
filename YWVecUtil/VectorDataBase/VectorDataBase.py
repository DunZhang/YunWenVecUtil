import numpy as np
import logging


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
            logging.info("no faiss in system")
            return
        logging.info("build faiss index...")
        if index_type == "IP":
            self.faiss_index = faiss.IndexFlatIP(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        elif index_type == "L2":
            self.faiss_index = faiss.IndexFlatL2(self.vector.shape[1])
            self.faiss_index.add(self.vector)
        logging.info("finish building faiss index")
