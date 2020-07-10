 
import numpy as np

import collections

import torch 

from sklearn.neighbors import NearestNeighbors,KDTree
def consolidate_caption_tuples(minibatch_list, outputs_list, opts, embedding_type='text'):
    """
    From a list of tuples which each have the form: 
    (caption, category, model_id, caption_embedding) 
    """
    caption_tuples = []
    seen_text = []
    seen_shapes = []

    
            	
    """
    first add all model_id's and text embeddings as tuple.
    """
    
    for minibatch, outputs in zip(minibatch_list, outputs_list):
        captions_tensor = minibatch['raw_embedding_batch']
        
        model_list = minibatch['model_list']
        for i in range(captions_tensor.shape[0]):
            
            caption = captions_tensor[i]
           
            #category = category_list[i]
            model_id = model_list[i//2]     

            caption_embedding_as_tuple = tuple(caption.tolist())
            if(caption_embedding_as_tuple in seen_text):
            	continue
            caption_embedding = outputs['text_encoder'][i]
            seen_text.append(caption_embedding_as_tuple)
            caption_tuple = (model_id, caption_embedding)#before model_id category,
            caption_tuples.append(caption_tuple)
            



    for minibatch, outputs in zip(minibatch_list, outputs_list):
        model_list = minibatch['model_list']
        for i in range(len(model_list)):#captions_tensor.shape[0]
            
            
           
            #category = category_list[i]
            model_id = model_list[i]
            
            #perhaps some rework needed such that we get all text and shape embeddings without skipping some of them
            if (model_id in seen_shapes):
            	continue
            	
            		
            
            shape_embedding = outputs['shape_encoder'][i]
            seen_shapes.append(model_id)
            caption_tuple = (model_id, shape_embedding)#category,
            caption_tuples.append(caption_tuple)
            

            
            
            #caption_tuple = (caption, category, model_id, shape_embedding)
            #caption_tuples.append(caption_tuple)

    

    return caption_tuples

def compute_metrics(embeddings_dict,n_neighbors = 10):
    
    (embeddings_matrix, labels, num_embeddings) = construct_embeddings_matrix(embeddings_dict)



    ##############################################################################################################
    ## in the function, we will use numpy
    ##############################################################################################################
    embeddings_matrix = embeddings_matrix.data.numpy() 
    labels = labels.data.numpy().astype(np.int32)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(embeddings_matrix)
    distances, indices = nbrs.kneighbors(embeddings_matrix)
    ind = []
    #print(embeddings_matrix)
    for i,ele in enumerate(indices):
   		ind.append(list(ele))
   		ind[i].remove(i)
   		indices = ind
   		

   	
    pr_at_k = compute_pr_at_k(indices, labels, n_neighbors, num_embeddings)
    #print(indices)
    

    return pr_at_k


def construct_embeddings_matrix(embeddings_dict):
    """
    Construct the embeddings matrix, which is NxD where N is the number of embeddings and D is
    the dimensionality of each embedding.
    Args:
        dataset: String specifying the dataset (e.g. 'synthetic' or 'shapenet')
        embeddings_dict: Dictionary containing the embeddings. It should have keys such as
                the following: ['caption_embedding_tuples', 'dataset_size'].
                caption_embedding_tuples is a list of tuples where each tuple can be decoded like
                so: caption, category, model_id, embedding = caption_tuple.
    """
    
    embedding_sample = embeddings_dict['caption_embedding_tuples'][0][1]
    embedding_dim = embedding_sample.shape[0]
    num_embeddings = embeddings_dict['dataset_size']
    

    
    embeddings_matrix = torch.zeros((num_embeddings, embedding_dim))
    labels = torch.zeros((num_embeddings))

    
    model_id_to_label = {}
    label_to_model_id = {}
    label_counter = 0
    
    

    for idx, caption_tuple in enumerate(embeddings_dict['caption_embedding_tuples']):

        # Parse caption tuple
        model_id, embedding = caption_tuple

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

        
    return embeddings_matrix, labels, num_embeddings,




def compute_pr_at_k(indices, labels, n_neighbors, num_embeddings, fit_labels=None):
   
    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

    #recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    
    Metrics = collections.namedtuple('Metrics',  'precision recall')# recall_rate ndcg')
    return Metrics(precision_at_k, recall_at_k)#, recall_rate_at_k, ave_ndcg_at_k)
