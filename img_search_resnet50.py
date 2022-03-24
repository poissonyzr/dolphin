import numpy as np
import os
from pathlib import Path
import pymilvus as milvus
from towhee import pipeline
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
from pymilvus import utility

embedding_pipeline = pipeline('towhee/image-embedding-resnet50')
dataset_path = '/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/train_images'

images = []
vectors = []


collection_name = 'reverse_image_search'
vec_dim = 2048
# connect to local Milvus service
milvus.connections.connect(host='127.0.0.1', port=19530)

# create collection
utility.drop_collection("reverse_image_search")
id_field = milvus.FieldSchema(name="id", dtype=milvus.DataType.INT64, is_primary=True, auto_id=True)
vec_field = milvus.FieldSchema(name="vec", dtype=milvus.DataType.FLOAT_VECTOR, dim=vec_dim)
schema = milvus.CollectionSchema(fields=[id_field, vec_field])
collection = milvus.Collection(name=collection_name, schema=schema)


for img_path in Path(dataset_path).glob('*'):
    time1=time.time()
    vec = embedding_pipeline(str(img_path))
    norm_vec = vec / np.linalg.norm(vec)
    vectors.append(norm_vec.tolist())
    images.append(str(img_path.resolve()))
    time2=time.time()
    print(time2-time1)


# insert data to Milvus
res = collection.insert([vectors])
collection.load()
img_dict = {}

# maintain mappings between primary keys and the original images for image retrieval
for i, key in enumerate(res.primary_keys):
    img_dict[key] = images[i]
np.save(r'/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/dict',img_dict)
query_img_path = '/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/test_images'
query_images = []
query_vectors = []
top_k = 5

for img_path in Path(query_img_path).glob('*'):
    vec = embedding_pipeline(str(img_path))
    norm_vec = vec / np.linalg.norm(vec)
    query_vectors.append(norm_vec.tolist())
    query_images.append(str(img_path.resolve()))

query_results = collection.search(data=query_vectors, anns_field="vec", param={"metric_type": 'L2'}, limit=top_k)

results_np=np.array(query_results)
np.save(r'/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/results',results_np)
query_images_np=np.array(query_images)
np.save(r'/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/query_images',query_images_np)

import numpy as np

dict_path='/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/dict.npy'
img_dict=np.load(dict_path,allow_pickle=True)

results_path='/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/results.npy'
result_lists=np.load(results_path,allow_pickle=True)

for result_list in result_lists:
    for result in result_list:
        try:
           
           print(img_dict.item()[result.id].split('/')[-1])
        except:
           print(result.id)
           print('no id')
