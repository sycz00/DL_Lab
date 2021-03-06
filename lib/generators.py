import numpy as np
from lib.utils import load_voxel
import sys
sys.path.append("..")
from config import cfg
import pickle
#04379243 cat_one TABLE
            #03001627 cat_two CHAIR

            #if(cur_category == '04379243'):
            #    db_ind = np.random.randint(self.num_data)
            #    continue
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

        #04379243 CLASS : TABLE
        #03001627 CLASS : CHAIR
        #if(cur_tup[1] == '04379243'):
            #continue

        #cur_caption_matches_id = val_inputs_dict['caption_matches'][cur_tup[2]]
        #print(len(cur_caption_matches_id))
        
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
    

def SS_generator(val_inputs_dict, opts):
    
    new_tuples = []
    seen_shapes = []
    
    probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
    with open(probablematic_nrrd_path, 'rb') as f: 
        bad_model_ids = pickle.load(f)

    for cur_tup in val_inputs_dict['caption_tuples']:
        if(cur_tup[2] in bad_model_ids):
            continue

        #04379243 CLASS : TABLE
        #03001627 CLASS : CHAIR
        #if(cur_tup[1] == '04379243'):
            #continue
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

