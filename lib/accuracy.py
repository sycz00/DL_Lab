 
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

    # print('Using unnormalized cosine distance')

    # Argsort method
    # unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    # sort_indices = np.argsort(unnormalized_similarities, axis=1)
    # # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    # indices = sort_indices[:, -n_neighbors:]
    # indices = np.flip(indices, 1)

    # Argpartition method
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
    print("TEST FOR ",n_neighbors)
    num_correct = 0
    all_counts = 0
    bb = []
    for emb in range(num_embeddings):
        
        label = labels[emb] #e.g. 0
        
        #counts = 0
        
        #counter = 0
        counts = sum(labels[indices[emb]] == label)
        #num_correct = 0
        #print("ind :",indices[emb])
        """
        for n in range(len(indices[emb])):
            neigh = indices[emb][n]
            if(neigh == emb):
                continue
            
            if(labels[neigh] == label):
                num_correct += 1
                counts += 1
                #break

            counter += 1
        
        """
        if(counts >= num_true):
            all_counts += 1

        #bb.append(num_correct/n_neighbors)
            
    
    return all_counts / num_embeddings#np.mean(bb)#

def compute_metrics(dataset, embeddings_dict, n_neighbors=20,nm=1,metric='minkowski', concise=False):
    """Compute all the metrics for the text encoder evaluation.
    """
    # assert len(embeddings_dict['caption_embedding_tuples']) < 10000
    # Dont need two sort steps!! https://stackoverflow.com/questions/1915376/is-pythons-sorted-function-guaranteed-to-be-stable
    # embeddings_dict['caption_embedding_tuples'] = sorted(embeddings_dict['caption_embedding_tuples'], key=lambda tup: tup[2])
    # embeddings_dict['caption_embedding_tuples'] = sorted(embeddings_dict['caption_embedding_tuples'], key=lambda tup: tup[0].tolist()) 
    #num_embeddings, label_to_model_idmodel_id_to_label
    (embeddings_matrix, labels, num_embeddings, label_counter) = construct_embeddings_matrix(embeddings_dict)#dataset,
    #print("LABELS:",labels[labels == 0])
    #print("label length",labels.shape)
    #print("label counter ",label_counter)
    #print('min embedding val:', np.amin(embeddings_matrix))
    #print('max embedding val:', np.amax(embeddings_matrix))
    #print('mean embedding (abs) val:', np.mean(np.absolute(embeddings_matrix)))

    n_neighbors = n_neighbors

    
    indices = _compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)

    print('Computing precision recall.')
    recall = acc_test(indices,labels,n_neighbors,num_embeddings,num_true=nm)
    #pr_at_k = compute_pr_at_k(indices, labels, n_neighbors, num_embeddings)


    # plot_pr_curve(pr_at_k)

    # Print some nearest neighbor indexes and labels (for debugging)
    # for i in range(10):
    #     print('Label:', labels[i])
    #     print('Neighbor indices:', indices[i][:5])
    #     print('Neighbors:', [labels[x] for x in indices[i][:5]])

    
    return recall
    #return pr_at_k  


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

        # Parse caption tuple
        model_id, embedding = caption_tuple#model_id, embedding = caption_tuple

        # Swap model ID and category depending on dataset
    
        # Add model ID to dict if it has not already been added
        #We ASSUME EACH MODEL_ID (WHICH IS A SPECIFIC SHAPE) BELONG TO ITS OWN INSTANCE-LEVEL label
        if model_id not in model_id_to_label:
            model_id_to_label[model_id] = label_counter
            label_to_model_id[label_counter] = model_id
            label_counter += 1

        # Update the embeddings matrix and labels vector
        embeddings_matrix[idx] = embedding
        labels[idx] = model_id_to_label[model_id]

        
    return embeddings_matrix, labels, num_embeddings,label_counter

"""
def construct_embeddings_matrix(dataset, embeddings_dict, model_id_to_label=None,
                                label_to_model_id=None):
    
    assert (((model_id_to_label is None) and (label_to_model_id is None)) or
            ((model_id_to_label is not None) and (label_to_model_id is not None)))
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][3]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = embeddings_dict['dataset_size']
    if (dataset == 'shapenet') and (num_embeddings > 30000):
        raise ValueError('Too many ({}) embeddings. Only use up to 30000.'.format(num_embeddings))
    assert embedding_sample.ndim == 1

    # Print info about embeddings
    print('Number of embeddings:', num_embeddings)
    print('Dimensionality of embedding:', embedding_dim)
    print('Estimated size of embedding matrix (GB):',
          embedding_dim * num_embeddings * 4 / 1024 / 1024 / 1024)
    print('')

    # Create embeddings matrix (n_samples x n_features) and vector of labels
    embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
    labels = np.zeros((num_embeddings)).astype(int)

    if (model_id_to_label is None) and (label_to_model_id is None):
        model_id_to_label = {}
        label_to_model_id = {}
        label_counter = 0
        new_dicts = True
    else:
        new_dicts = False

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        # Parse caption tuple
        caption,category, model_id, embedding = caption_tuple

        # Swap model ID and category depending on dataset
        if dataset == 'primitives':
            tmp = model_id
            model_id = category
            category = tmp

        # Add model ID to dict if it has not already been added
        if new_dicts:
            if model_id not in model_id_to_label:
                model_id_to_label[model_id] = label_counter
                label_to_model_id[label_counter] = model_id
                label_counter += 1

        # Update the embeddings matrix and labels vector
        embeddings_matrix[idx] = embedding
        labels[idx] = model_id_to_label[model_id]

        # Print progress
        if (idx + 1) % 10000 == 0:
            print('Processed {} / {} embeddings'.format(idx + 1, num_embeddings))
    return embeddings_matrix, labels, model_id_to_label, num_embeddings, label_to_model_id
"""

def compute_pr_at_k(indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)

    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    #print('recall_at_k shape:', recall_at_k.shape)
    print('     k: precision recall recall_rate ndcg')
    for k in range(n_neighbors):
        print('pr @ {}: {} {} {} {}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k]))
    Metrics = collections.namedtuple('Metrics', 'precision recall recall_rate ndcg')
    return Metrics(precision_at_k, recall_at_k, recall_rate_at_k, ave_ndcg_at_k)

def print_model_id_info(model_id_to_label):
    print('Number of models (or categories if synthetic dataset):', len(model_id_to_label.keys()))
    print('')

    # Look at a few example model IDs
    
def print_nearest_info(query_model_ids, nearest_model_ids, query_sentences, nearest_sentences,
                       render_dir=None):
    """Print out nearest model IDs for random queries.

    Args:
        labels: 1D array containing the label
    """
    # Make directory for renders
    if render_dir is None:
        render_dir = os.path.join('/tmp', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(render_dir)

    num_queries = 25
    assert len(nearest_model_ids) > num_queries
    perm = np.random.permutation(len(nearest_model_ids))
    for i in perm[:num_queries]:
        query_model_id = query_model_ids[i]
        nearest = nearest_model_ids[i]

        # Make directory for the query
        cur_render_dir = os.path.join(render_dir, query_model_id + ('-%04d' % i))
        os.makedirs(cur_render_dir)

        with open(os.path.join(cur_render_dir, 'nearest_neighbor_text.txt'), 'w') as f:
            f.write('-------- query {} ----------\n'.format(i))
            f.write('Query: {}\n'.format(query_model_id))
            f.write('Nearest:\n')
            for model_id in nearest:
                f.write('\t{}\n'.format(model_id))
            render.render_model_id([query_model_id] + nearest, out_dir=cur_render_dir, check=False)

            f.write('')
            query_sentence = query_sentences[i]
            f.write('Query: {}\n'.format(query_sentence))
            for sentence in nearest_sentences[i]:
                f.write('\t{}\n'.format(sentence))
            f.write('')

        ids_only_fname = os.path.join(cur_render_dir, 'ids_only.txt')
        with open(ids_only_fname, 'w') as f:
            f.write('{}\n'.format(query_model_id))
            for model_id in nearest:
                f.write('{}\n'.format(model_id))
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


