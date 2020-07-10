import os
import logging

logging.basicConfig(level=logging.INFO)
PROJ_PATH = r"D:\Codes\PythonProj\YunWenVecUtil"
import sys

sys.path.append(os.path.join(PROJ_PATH, "SentenceEncoder"))
sys.path.append(os.path.join(PROJ_PATH, "VectorDataBase"))
sys.path.append(os.path.join(PROJ_PATH, "VectorSearch"))
import torch
from WVSentenceEncoder import WVSentenceEncoder
from BERTSentenceEncoder import BERTSentenceEncoder
from VectorSearchUtil import find_topk_by_sens, find_topk_by_vecs
from VectorDataBase import VectorDataBase

sens = ["今天天气不错哦", "外面下了好大的雨", "测试句向量库"]
sens_words = [["今天", "天气", "不错", "哦"],
              ["外面", "下了", "好大", "的", "雨"],
              ["测试", "句", "向量", "库"]]
### 句编码器路径
wv_path = r"D:\谷歌下载目录\Tencent_AILab_ChineseEmbedding.txt"
bert_dir = r"D:\Codes\PretrainedModel\simbert_torch"
### 句编码器
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wv_encoder = WVSentenceEncoder(wv_path, binary=False, limit=10000)
bert_encoder = BERTSentenceEncoder(bert_dir, device, pooling_modes=["cls", "max"], batch_size=8)
### 目标数据库
vec_db_bert = VectorDataBase(bert_encoder.get_sens_vec(sens))
vec_db_wv = VectorDataBase(wv_encoder.get_sens_vec(sens_words))
print("vec_db_bert shape", vec_db_bert.vector.shape)
print("vec_db_wv shape", vec_db_wv.vector.shape)
### 搜索
source_vec_wv = vec_db_wv.vector[0:1, :]
source_vec_bert = vec_db_bert.vector[0:1, :]

# res_index, res_distance = find_topk_by_vecs(source_vec_bert, vec_db_bert, topk=2, metric="cosine", use_faiss=True)
# print(res_index, res_distance)
# res_index, res_distance = find_topk_by_vecs(source_vec_bert, vec_db_bert, topk=2, metric="cosine", use_faiss=False)
# print(res_index, res_distance)
res_index, res_distance = find_topk_by_vecs(source_vec_wv, vec_db_wv, topk=2, metric="cosine", use_faiss=True)
print(res_index, res_distance)
# res_index, res_distance = find_topk_by_vecs(source_vec_wv, vec_db_wv, topk=2, metric="cosine", use_faiss=False)
# print(res_index, res_distance)
# res = find_topk_by_sens(wv_encoder, sens_words[0:1], sens_words, topk=2, metric="cosine", use_faiss=False)
# print(res)
# res = find_topk_by_sens(bert_encoder, sens[0:1], sens, topk=2, metric="euclidean", use_faiss=True)
# print(res)
