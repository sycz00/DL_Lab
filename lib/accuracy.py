 
import numpy as np

import collections

import torch 

from sklearn.neighbors import NearestNeighbors,KDTree
from lib.utils import load_voxel


#generates output for caption and shape embeddings
def TS_generator(val_inputs_dict, opts):
    
    new_tuples = []
    seen_captions = []
	#label_counter = 0
    for cur_tup in val_inputs_dict['caption_tuples']:
        
        cur_caption = tuple(cur_tup[0].tolist())
        if cur_caption not in seen_captions:
            seen_captions.append(cur_caption)
            cur_model_id = cur_tup[2]
            cur_shape = load_voxel(None, cur_model_id, opts)
            new_tuples.append((cur_model_id,cur_caption,cur_shape))
			


    caption_tuples = new_tuples
    raw_caption_list = [tup[1] for tup in caption_tuples]
    raw_shape_list = [tup[2] for tup in caption_tuples]
    model_list = [tup[0] for tup in caption_tuples]
    n_captions = len(raw_caption_list)
    n_loop_captions = n_captions - (n_captions % opts.batch_size)
    print('number of captions: {0}'.format(n_captions))
    print('number of captions to loop through for validation: {0}'.format(n_loop_captions))
    print('number of batches to loop through for validation: {0}'.format(n_loop_captions/opts.batch_size))
    for start in range(0, n_loop_captions, opts.batch_size):
        captions = raw_caption_list[start:(start + opts.batch_size)]
        shapes = raw_shape_list[start:(start + opts.batch_size)]
        minibatch = {
        'raw_embedding_batch': np.asarray(captions),
        'voxel_tensor_batch': np.array(shapes).astype(np.float32) ,
        'model_list': model_list[start:(start + opts.batch_size)]
        }
        yield minibatch

def S_generator(val_inputs_dict, opts):
    new_tuples = []
    seen_shapes = []
    k = []
    matches_keys = list(val_inputs_dict['caption_matches'].keys())
    for match_key in matches_keys:
        cur_caption_matches_id = val_inputs_dict['caption_matches'][match_key]
        if(len(cur_caption_matches_id) < 4):
            continue
        #if(len(cur_caption_matches_id) < 4):
        #    continue
        #print("NEW")
        for i_d in cur_caption_matches_id[0:4]:
            
            tup = val_inputs_dict['caption_tuples'][i_d]
            cur_model_id = tup[2]
            if(cur_model_id in seen_shapes):
                continue
            
            seen_shapes.append(cur_model_id)
            #cur_caption = tup[0]
            #cur_model_id = tup[2]
            #print("cur model id:",cur_model_id)
            cur_shape = load_voxel(None, cur_model_id, opts)
            new_tuples.append((cur_model_id,cur_shape))
        #input()
    caption_tuples = new_tuples
    #raw_caption_list = [tup[1] for tup in caption_tuples]
    model_list = [tup[0] for tup in caption_tuples]
    raw_shape_list = [tup[1] for tup in caption_tuples]
    n_captions = len(raw_shape_list)
    n_loop_captions = n_captions - (n_captions % opts.batch_size)
    print('number of SHAPES: {0}'.format(n_captions))
    print('number of SHAPES to loop through for validation: {0}'.format(n_loop_captions))
    print('number of batches to loop through for validation: {0}'.format(n_loop_captions/opts.batch_size))
    for start in range(0, n_loop_captions, opts.batch_size):
        #captions = raw_caption_list[start:(start + opts.batch_size)]
        shapes = raw_shape_list[start:(start + opts.batch_size)]
        minibatch = {
        #'raw_embedding_batch': np.asarray(captions),
        'voxel_tensor_batch': np.array(shapes).astype(np.float32),
        'model_list': model_list[start:(start + opts.batch_size)]
        }
        yield minibatch
#generates outputs for only caption matches.
def TT_generator(val_inputs_dict, opts):
    new_tuples = []
    seen_captions = []
    k = []
    matches_keys = list(val_inputs_dict['caption_matches'].keys())
    for match_key in matches_keys:
        cur_caption_matches_id = val_inputs_dict['caption_matches'][match_key]
        if(len(cur_caption_matches_id) < 4):
            continue
        #if(len(cur_caption_matches_id) < 4 ):
            #continue
        #print("NEW")
        for i_d in cur_caption_matches_id[0:4]:
            
            tup = val_inputs_dict['caption_tuples'][i_d]
            cur_caption = tuple(tup[0].tolist())
            if(cur_caption in seen_captions):
                continue
            seen_captions.append(cur_caption)
            #cur_caption = tup[0]
            cur_model_id = tup[2]
            #print("cur model id:",cur_model_id)
            #cur_shape = load_voxel(None, cur_model_id, opts)
            new_tuples.append((cur_model_id,cur_caption))
        #input()
    caption_tuples = new_tuples
    raw_caption_list = [tup[1] for tup in caption_tuples]
    model_list = [tup[0] for tup in caption_tuples]
    #raw_shape_list = [tup[2] for tup in caption_tuples]
    n_captions = len(raw_caption_list)
    n_loop_captions = n_captions - (n_captions % opts.batch_size)
    print('number of captions: {0}'.format(n_captions))
    print('number of captions to loop through for validation: {0}'.format(n_loop_captions))
    print('number of batches to loop through for validation: {0}'.format(n_loop_captions/opts.batch_size))
    for start in range(0, n_loop_captions, opts.batch_size):
        captions = raw_caption_list[start:(start + opts.batch_size)]
        #shapes = raw_shape_list[start:(start + opts.batch_size)]
        minibatch = {
		'raw_embedding_batch': np.asarray(captions),
        #'voxel_tensor_batch': np.array(shapes).astype(np.float32),
		'model_list': model_list[start:(start + opts.batch_size)]
		}
        yield minibatch


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
    # query_embeddings_matrix: 3000 x 128 
    # fit_embeddings_matrix: 17000 x 128
    # resulted unnormalized_similarities: 3000 x 17000
    unnormalized_similarities = np.dot(query_embeddings_matrix, fit_embeddings_matrix.T)
    #unnormalized_similarities /= 128
    #np.savetxt('sim.txt',unnormalized_similarities,fmt='%.2f')
    
    n_samples = unnormalized_similarities.shape[0]
    ################################################################################################
    # np.argpartition: It returns an array of indices of the same shape as a that
    #   index data along the given axis in partitioned order.
    # kth : int or sequence of ints, Element index to partition by. The k-th element will be in its final sorted position and all smaller elements will
    # be moved before it and all larger elements behind it. The order all elements in the partitions is
    # undefined. If provided with a sequence of k-th it will partition all of them into their sorted 
    # position at once.
    #################################################################################################
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    # -n_neighbors is in its position, all values bigger than sort_indices[-n_neighbors]
    # is on the right
    indices = sort_indices[:, -n_neighbors:]
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ...., 29999, .., 2999]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)] #[0,0,1,1,2,2,3,3,...]
    # take out nearest n_neighbors elements
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

def Dot_Similarity(x, y):
    if len(x) != len(y):
        raise ValueError("x and y need to have the same length")
    return np.dot(x,y.T)
    #return math.sqrt(sum([(y[i] - x[i]) ** 2 for i in range(len(x))]))

def compute_metrics(embeddings_dict,n_neighbors = 10):
    
    (embeddings_matrix, labels, num_embeddings,label_counter) = construct_embeddings_matrix(embeddings_dict)



    ##############################################################################################################
    ## in the function, we will use numpy
    ##############################################################################################################
    embeddings_matrix = embeddings_matrix
    labels = labels#.astype(np.int32)
    
    #tt = np.dot(embeddings_matrix,embeddings_matrix.T)
    #print(labels[0:10])
    #print(labels[5840:5860])
    indices = _compute_nearest_neighbors_cosine(embeddings_matrix,embeddings_matrix,n_neighbors,True)
    #indices = simple_text_NN(embeddings_matrix,n_neighbors)
    #nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(embeddings_matrix)}).fit(embeddings_matrix)
    #nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(embeddings_matrix)
    
    #nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto',metric=Dot_Similarity).fit(embeddings_matrix)
    
    #distances, indices = nbrs.kneighbors(embeddings_matrix)
   

    #pr = compute_pr_at_k(indices, labels, n_neighbors, num_embeddings)
    
    pr = acc_test(indices, labels, n_neighbors, num_embeddings)
   
    return pr


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
    

    
    embeddings_matrix = np.zeros((num_embeddings, embedding_dim))
    labels = np.zeros((num_embeddings))

    
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

        
    return embeddings_matrix, labels, num_embeddings,label_counter


def acc_test(indices,labels,n_neighbors,num_embeddings):
	num_correct = 0
	all_counts = 0
	bb = []
	for emb in range(num_embeddings):
		#print("Embedd",emb)
		label = labels[emb]
		#print("Current label",label)
		#print("indices : ",indices[emb])
		num_correct = 0
		counter = 0
		for n in range(len(indices[emb])):
			neigh = indices[emb][n]
			if(neigh == emb):
				continue
			#print("Neighbor :",neigh)
			#print("label :",labels[neigh])
			if(labels[neigh] == label):
				num_correct += 1

			counter += 1
		
		#print(num_correct/counter)
		#input()
		bb.append(num_correct/counter)
			
	#print(num_correct)
	#print(num_correct / all_counts)
	#print("END",np.mean(bb))
	#input()
	return np.mean(bb)#num_correct/all_counts


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

"""

def consolidate_caption_tuples(minibatch_list, outputs_list, opts, embedding_type='text'):
   
    caption_tuples = []
    seen_text = []
    seen_shapes = []

    
            	
   
    if(embedding_type =='text'):
    	for minibatch, outputs in zip(minibatch_list, outputs_list):
        	captions_tensor = minibatch['raw_embedding_batch']
        
        	model_list = minibatch['model_list']
        	for i in range(captions_tensor.shape[0]):
        		caption = captions_tensor[i]
        		model_id = model_list[i]
        		caption_embedding_as_tuple = tuple(caption.tolist())
        		if(caption_embedding_as_tuple in seen_text):
        			continue
        		caption_embedding = outputs['text_encoder'][i]
        		seen_text.append(caption_embedding_as_tuple)
        		caption_tuple = (model_id, caption_embedding)
        		caption_tuples.append(caption_tuple)
            
    else:
       
    	
    	for minibatch, outputs in zip(minibatch_list, outputs_list):
        	model_list = minibatch['model_list']
        	for i in range(len(model_list)//2):
        		model_id = model_list[int(i*2)]

        		if (model_id in seen_shapes):
        			continue
        		shape_embedding = outputs['shape_encoder'][i]
        		seen_shapes.append(model_id)
        		caption_tuple = (model_id, shape_embedding)
        		caption_tuples.append(caption_tuple)
            

            
            
            

    #np.savetxt('id_2.txt',np.array(seen_shapes))  

    return caption_tuples
 """