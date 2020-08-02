import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image
import utils as utils
import sys
sys.path.append("..")
from config import cfg
import matplotlib.image as mpimg

def plot_TSNE(Embeddings,Shapes):

	



	tsne = TSNE().fit_transform(Embeddings)#n_components=2, random_state=0
	tx, ty = tsne[:,0], tsne[:,1]
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
	width = 4000
	height = 3000
	max_dim = 100
	full_image = Image.new('RGB', (width, height))
	for idx, x in enumerate(Shapes):
		tile = Image.fromarray(np.uint8(x * 255))
		rs = max(1, tile.width / max_dim, tile.height / max_dim)
		tile = tile.resize((int(tile.width / rs),int(tile.height / rs)),Image.ANTIALIAS)
		full_image.paste(tile, (int((width-max_dim) * tx[idx]),int((height-max_dim) * ty[idx])))
	full_image.show()


if __name__ == "__main__":
	
	

	embeddings_trained = utils.open_pickle('../EMBEDDINGS/METRIC_ONLY/text_and_shape.p')
	embeddings, model_ids = utils.create_embedding_tuples(embeddings_trained,embedd_type='text__')
	model_ids = np.array(model_ids)

	embeddings_trained_2 = utils.open_pickle('../EMBEDDINGS/METRIC_TST/text_and_shape.p')
	embeddings_2, model_ids_2 = utils.create_embedding_tuples(embeddings_trained_2,embedd_type='text__')
	model_ids_2 = np.array(model_ids_2)

	indices = np.random.choice(np.arange(len(embeddings)),size=150,replace=False)
	assert len(embeddings) == len(embeddings_2)
	#print(indices)
	embeddings = embeddings[indices]
	embeddings_2 = embeddings_2[indices]
	#print(embeddings)
	model_ids = model_ids[indices]
	model_ids_2 = model_ids_2[indices]

	
	Shapes = []
	for id_ in model_ids:
		pic = cfg.DIR.RGB_PNG_PATH % (id_,id_)
		img = mpimg.imread(pic)
		Shapes.append(img)
	plot_TSNE(embeddings,Shapes)

	input()
	Shapes = []
	for id_ in model_ids_2:
		pic = cfg.DIR.RGB_PNG_PATH % (id_,id_)
		img = mpimg.imread(pic)
		Shapes.append(img)
	plot_TSNE(embeddings_2,Shapes)
        

