import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dict_path='/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/dict.npy'
img_dict=np.load(dict_path,allow_pickle=True)
query_images=np.load('/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/query_images.npy')
query_results=np.load('/home/zilliz/Downloads/Kaggle/happy-whale-and-dolphin/results.npy',allow_pickle=True)

for i in range(5):
    results = query_results[i]
    query_file = query_images[i]
    result_files = [img_dict.item()[result.id] for result in results]
    distances = [result.distance for result in results]

    fig_query, ax_query = plt.subplots(1,1, figsize=(5,5))
    ax_query.imshow(Image.open(query_file))
    ax_query.set_title("Searched Image\n")
    ax_query.axis('off')

    fig, ax = plt.subplots(1,len(result_files),figsize=(20,20))
    for x in range(len(result_files)):
        ax[x].imshow(Image.open(result_files[x]))
        ax[x].set_title('dist: ' + str(distances[x])[0:5])
        ax[x].axis('off')
    plt.show()
