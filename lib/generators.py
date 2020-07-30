import numpy as np
from lib.utils import load_voxel
import sys
sys.path.append("..")
from config import cfg
import pickle

def TS_generator(val_inputs_dict, opts):
    
    new_tuples = []
    seen_captions = []
	#label_counter = 0
    probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
    with open(probablematic_nrrd_path, 'rb') as f: 
        bad_model_ids = pickle.load(f)

    for cur_tup in val_inputs_dict['caption_tuples']:
        
        cur_caption = tuple(cur_tup[0].tolist())
        if(cur_tup[2] in bad_model_ids):
            continue
        if cur_caption not in seen_captions:
            
            seen_captions.append(cur_caption)
            cur_model_id = cur_tup[2] #changed it to category instead of model id. model is is cur_tup[2]
            #cur_shape = load_voxel(None, cur_model_id, opts)
            new_tuples.append((cur_model_id,cur_caption))#,cur_shape))
			


    caption_tuples = new_tuples
    raw_caption_list = [tup[1] for tup in caption_tuples]
    #raw_shape_list = [tup[2] for tup in caption_tuples]
    model_list = [tup[0] for tup in caption_tuples]
    return raw_caption_list, model_list
    """
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
        #'voxel_tensor_batch': np.array(shapes).astype(np.float32) ,
        'model_list': model_list[start:(start + opts.batch_size)]
        }
        yield minibatch
    """

def SS_generator(val_inputs_dict, opts):
    
    new_tuples = []
    seen_shapes = []
    
    probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
    with open(probablematic_nrrd_path, 'rb') as f: 
        bad_model_ids = pickle.load(f)

    for cur_tup in val_inputs_dict['caption_tuples']:
        if(cur_tup[2] in bad_model_ids):
            continue
        cur_model_id = cur_tup[2]
        if cur_model_id not in seen_shapes:
            
            seen_shapes.append(cur_model_id)
            cur_model_id = cur_tup[2] #changed it to category instead of model id. model is is cur_tup[2]
            #cur_category = cur_tup[1]
            cur_shape = load_voxel(None, cur_model_id, opts)
            new_tuples.append((cur_model_id,cur_shape))#,cur_shape))
            
            


    caption_tuples = new_tuples
    #raw_caption_list = [tup[1] for tup in caption_tuples]
    raw_shape_list = [tup[1] for tup in caption_tuples]
    model_list = [tup[0] for tup in caption_tuples]

    return raw_shape_list, model_list
    #category_list = [tup[0] for tup in caption_tuples]

def generator_minibatch(input_list, model_list, opts):
    n_captions = len(input_list)
    n_loop_captions = n_captions - (n_captions % opts.batch_size) +1
    print('number of SHAPES: {0}'.format(n_captions))
    print('number of captions to loop through for validation: {0}'.format(n_loop_captions))
    print('number of batches to loop through for validation: {0}'.format(n_loop_captions/opts.batch_size))

    for start in range(0, n_loop_captions, opts.batch_size):
        inputs = input_list[start:(start + opts.batch_size)]
        minibatch = {
        'input': np.array(inputs).astype(np.float32) ,
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
        #if(len(cur_caption_matches_id) < 4):
        #    continue
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