 
import numpy as np

import collections

import torch 

from sklearn.neighbors import NearestNeighbors,KDTree
from lib.utils import load_voxel,convert_idx_to_words
import torch.nn.functional as F
from sklearn.preprocessing import normalize
###############
import sys
sys.path.append("..")
from config import cfg
import pickle

#generates output for caption and shape embeddings



def _compute_nearest_neighbors_cosine(fit_embeddings_matrix, query_embeddings_matrix,
                                      n_neighbors, fit_eq_query, range_start=0):
    if fit_eq_query is True:
        n_neighbors += 1

    

    # Argsort method
    # unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    # sort_indices = np.argsort(unnormalized_similarities, axis=1)
    # # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    # indices = sort_indices[:, -n_neighbors:]
    # indices = np.flip(indices, 1)

    # Argpartition method
    #query_embeddings_matrix = normalize(query_embeddings_matrix, axis=1)
    #fit_embeddings_matrix = normalize(fit_embeddings_matrix, axis=1)
     
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    n_samples = unnormalized_similarities.shape[0]
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    indices = sort_indices[:, -n_neighbors:]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)

    if fit_eq_query is True:
        n_neighbors -= 1  # Undo the neighbor increment
        final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
        compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
        has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
        any_result = np.any(has_self, axis=1)
        for row_idx in range(indices.shape[0]):
            if any_result[row_idx]:
                nonzero_idx = np.nonzero(has_self[row_idx, :])
                assert len(nonzero_idx) == 1
                new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
                final_indices[row_idx, :] = new_row
            else:
                final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
        indices = final_indices
    return indices




def acc_test(indices,labels,n_neighbors,num_embeddings,num_true=1):
    
    num_correct = 0
    all_counts = 0
    bb = []
    for emb in range(num_embeddings):
        
        label = labels[emb] #e.g. 0
        
        counts = sum(labels[indices[emb]] == label)
        
        
        if(counts >= num_true):
            all_counts += 1

    return all_counts / num_embeddings#np.mean(bb)#

def compute_metrics(dataset, embeddings_dict, n_neighbors=20,nm=1,metric='minkowski', concise=False):
    
    (embeddings_matrix, labels, num_embeddings, label_counter) = construct_embeddings_matrix(embeddings_dict)
    
    n_neighbors = n_neighbors

    indices = _compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)

    print('Computing recall.')
    recall = acc_test(indices,labels,n_neighbors,num_embeddings,num_true=nm)
    
    return recall
     


def construct_embeddings_matrix(embeddings_dict):
    
    
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][1]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = embeddings_dict['dataset_size']
    

    
    embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
    labels = np.zeros((num_embeddings))

    
    model_id_to_label = {}
    label_to_model_id = {}
    label_counter = 0
    
    

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):
        model_id, embedding = caption_tuple#model_id, embedding = caption_tuple

    
        # Add model ID to dict if it has not already been added
        #We ASSUME EACH MODEL_ID (WHICH IS A SPECIFIC SHAPE) BELONG TO ITS OWN INSTANCE-LEVEL label
        if model_id not in model_id_to_label:
            model_id_to_label[model_id] = label_counter
            label_to_model_id[label_counter] = model_id
            label_counter += 1

        
        embeddings_matrix[idx] = embedding
        labels[idx] = model_id_to_label[model_id]

        
    return embeddings_matrix, labels, num_embeddings,label_counter


################################################################################################################
def main():
    
    #DIR.SHAPENET_METRIC_EMBEDDINGS_VAL
    with open(cfg.DIR.SHAPENET_METRIC_EMBEDDINGS_TEST, 'rb') as f:
        embeddings_dict = pickle.load(f)

    
    np.random.seed(1234)
    #embeddings_dict = np.array(embeddings_dict)
    compute_metrics("shapenet",embeddings_dict)


if __name__ == '__main__':
    main()


