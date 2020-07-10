"""
使用该库的一个简单例子：获取一批句子的向量，并从中找个k个最相似的
"""
import os
import torch
import logging
logging.basicConfig(level=logging.INFO)
from sklearn.preprocessing import normalize
from YWVecUtil import BERTSentenceEncoder,VectorDataBase,find_topk_by_sens,find_topk_by_vecs

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### 拿着source去target里搜索
source_sens = ["魔幻手机第三部什么时候上映？", "抛货价什么意思"]

target_sens = ["魔幻手机第三部什么时候上映开播？", "上货价什么意思",
               "郑州割双眼皮一般需要多少钱", "去哪个网站买正品化妆品？",
               "告诉我关于异界女神的小说", "关于异界女神的小说告诉我"]*100
### simbert 句编码器
sen_encoder = BERTSentenceEncoder(r"D:\Codes\PretrainedModel\simbert_torch",
                                  device,
                                  pooling_modes=["cls"],
                                  batch_size=8)
### 为target生成句向量库
vec_db = VectorDataBase(sen_encoder.get_sens_vec(target_sens))
vec_db.vector = normalize(vec_db.vector, axis=1)  # 使用cosine 需要先进行单位化
vec_db.build_faiss_index(index_type="IP")  # 构建索引，faiss不可用时，会自动跳过
# vec_db.vector vector属性就是句向量了
### 开始搜索
# use_faiss="auto" faiss可用时会自动使用以加速
res = find_topk_by_sens(sen_encoder, source_sens, target_sens, topk=3,
                        metric="cosine", use_faiss="auto")
### 输出结果
for i in res:
    print(i)
