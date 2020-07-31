import os 
import nrrd 
import numpy as np
import pickle
import json  
import time 
import collections
import datetime 
import torch
from render import render_model_id #lib.render import render_model_id 
import sys
sys.path.append("..")
from config import cfg
#from config import cfg

import pdb 



def convert_idx_to_words(data_list):
    """Converts each sentence/caption in the data_list using the idx_to_word dict.
    Args:
        idx_to_word: A dictionary mapping word indices (keys) in string format (?) to words.
        data_list: A list of dictionaries. Each dictionary contains a 'raw_embedding' field (among
            other fields) that is a list of word indices.
    Returns:
        sentences: A list of sentences (strings).
    """
    inputs_list = json.load(open(cfg.DIR.JSON_PATH, 'r'))
    idx_to_word = inputs_list['idx_to_word']

    sentences = []
    #for idx, cur_dict in enumerate(data_list):
    #    sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)] for word_idx in cur_dict['raw_caption_embedding'] if word_idx != 0]))
    sentences.append(' '.join([idx_to_word[str(word_idx)] for word_idx in data_list if word_idx != 0]))

    return sentences

def create_embedding_tuples(trained_embeddings,embedd_type='text'): 
    #print(trained_embeddings.keys())
    dim_emb = trained_embeddings['dataset_size']
    embeddings_matrix = np.zeros((dim_emb, 128))
    cat_mod_id = []
    raw_caption = []
    for idx,entry in enumerate(trained_embeddings['caption_embedding_tuples']):
        
        embeddings_matrix[idx] = entry[1]
        cat_mod_id.append(entry[0])
        if(embedd_type == 'text'):
            raw_caption.append(entry[2])

    if(embedd_type == 'text'):
        return embeddings_matrix,cat_mod_id,raw_caption    
    else:
        return embeddings_matrix,cat_mod_id

def make_data_processes(data_process_class, queue, data_paths, opts, repeat): 
    
    processes = [] 
    for i in range(opts.num_workers): 
        process = data_process_class(queue, data_paths, opts, repeat=repeat)
        process.start() 
        processes.append(process) 

    return processes 

def kill_processes(queue, processes): 
    print('signal processes to shutdown')

    for p in processes: 
        p.shutdown() 

    
    
    while not queue.empty(): # If queue is not empty 
        time.sleep(0.5)
        try: 
            queue.get(False) 
        except:
            print('now queue size is {0}'.format(queue.qsize()))
            pass 

    print('killing processes.') 
    for p in processes:
        p.terminate() 

def create_pickle_embedding(mat,embedd_type ='shape'):
    print("start pickle !")
    dict_ = {}
    seen_models = []
    tuples = []
    for ele in mat:
        
        if(ele[0] in seen_models):
            continue
        else:
            seen_models.append(ele[0])
            if(embedd_type == 'text_only'):
                tuples.append((ele[0],ele[1],ele[2]))
            else:
                tuples.append((ele[0],ele[1]))
            
        
    
    dict_ = {
    'caption_embedding_tuples': tuples,
    'dataset_size':len(tuples)
    }

    output = open('{}.p'.format(embedd_type), 'wb')
    pickle.dump(dict_, output)
    output.close()


    print("created pickle file for {}".format(embedd_type))
# read nrrd data 
def read_nrrd(nrrd_filename):
    """
    Reads an NRRD file and returns an RGBA tensor 
    Args: 
        nrrd_filename: filename of nrrd file 
    Returns: 
        voxel tensor: 4-dimensional voxel tensor with 4 channels (RGBA) where the alpha channel 
            is the last channel(aka vx[:, :, :, 3]).
    """
    nrrd_tensor, options = nrrd.read(nrrd_filename)
    assert nrrd_tensor.ndim == 4 

    # convert to float [0,1]
    voxel_tensor = nrrd_tensor.astype(np.float32) / 255 
    # Move channel dimension to last dimensions 
    voxel_tensor = np.rollaxis(voxel_tensor, 0, 4) 

    # Make the model stand up straight by swapping axes
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 1) 
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 2) 

    return voxel_tensor
# write nrrd 
def write_one_voxel2nrrd(voxel_tensor, filename):
    """
    Converts binvox tensor to NRRD (RGBA) format and writes it if a filename is provided.
    Example usage:
        voxel_tensor = np.load('text2voxel/output/tmp/ckpt-10500/0000_voxel_tensor_output.npy')
        _ = nrrd_rw.write_nrrd(voxel_tensor, filename='./models_checkpoint/test_nrrd.nrrd')
    Args:
        voxel_tensor: A tensor representing the binary voxels. Values can range from 0 to 1, and
            they will be properly scaled. The format is [height, width, depth, channels].
        filename: Filename that the NRRD will be written to.
    Writes:
        nrrd_tensor: An RGBA tensor where the channel dimension (RGBA) comes first
            (channels, height, width, depth).
    """
    if voxel_tensor.ndim == 3:  # Add a channel if there is no channel dimension
        voxel_tensor = voxel_tensor[np.newaxis, :]
    elif voxel_tensor.ndim == 4:  # Roll axes so order is (channel, height, width, depth) (not sure if (h, w, d))
        voxel_tensor = np.rollaxis(voxel_tensor, 3)
    else:
        raise ValueError('Voxel tensor must have 3 or 4 dimensions.')

    # Convert voxel_tensor to uint8
    voxel_tensor = (voxel_tensor * 255).astype(np.uint8)

    if voxel_tensor.shape[0] == 1:  # Add channels if voxel_tensor is a binvox tensor
        nrrd_tensor_slice = voxel_tensor
        nrrd_tensor = np.vstack([nrrd_tensor_slice] * 4)
        nrrd_tensor[:3, :, :, :] = 128  # Make voxels gray
        nrrd_tensor = nrrd_tensor.astype(np.uint8)
    elif voxel_tensor.shape[0] == 4:
        nrrd_tensor = voxel_tensor
    elif voxel_tensor.shape[0] != 4:
        raise ValueError('Voxel tensor must be single-channel or 4-channel')

    nrrd.write(filename, nrrd_tensor)


def get_voxel_file(category, model_id, opts):
    """
    get the voxel absolute filepath for the model specified by category and model_id 
    Args: 
        category: category of the model as a string , e.g., '03001627'
        model_id: model id of the model as a string, e.g., '587ee5822bb56bd07b11ae648ea92233'
    Returns: 
        voxel_filepath: Filepath of the binvox file corresponding to category and model_id 
    """
    #if opts.dataset == 'shapenet': # shapenet dataset 
    return opts.data_dir % (model_id, model_id) 
    #elif opts.dataset == 'primitives': # primitives dataset

    #return opts.data_dir % (category, model_id) 
    #else: 
        #raise ValueError('please use a valid dataset (shapenet, primitives)')

def load_voxel(category, model_id, opts): 
    """
    Loads the voxel tensor given the model category and model ID 
    Args: 
        category: model category
        model_id: model id 
    Returns: 
        voxel tensor of shape (height x width x depth x channels) 
    """
    
    voxel_fn = get_voxel_file(category, model_id, opts)

    
    voxel_tensor = read_nrrd(voxel_fn) 
    
    return voxel_tensor 



def augment_voxel_tensor(voxel_tensor, max_noise=10):
    """
    Arguments the RGB values of the voxel tensor. The RGB channelss are perturbed by the same single
    noise value, and the noise is sampled from a uniform distribution.
    Args: 
        voxel_tensor: a single voxel tensor 
        max_noise: Integer representing max noise range. We will perform voxel_value + max_noise to 
        augment the voxel tensor, where voxel_value and max_noise are [0, 255].
    Returns: 
        augmented_voxel_tensor: voxel tensor after the data augmentation
    """
    augmented_voxel_tensor = np.copy(voxel_tensor) # do nothing if binvox 
    if (voxel_tensor.ndim == 4) and (voxel_tensor.shape[3] != 1) and (max_noise > 0):
        noise_val = float(np.random.randint(-max_noise, high=(max_noise + 1))) / 255
        augmented_voxel_tensor[:, :, :, :3] += noise_val
        augmented_voxel_tensor = np.clip(augmented_voxel_tensor, 0., 1.)
    return augmented_voxel_tensor

def rescale_voxel_tensor(voxel_tensor):
    """Rescales all values (RGBA) in the voxel tensor from [0, 1] to [-1, 1].
    Args:
        voxel_tensor: A single voxel tensor.
    Returns:
        rescaled_voxel_tensor: A single voxel tensor after rescaling.
    """
    rescaled_voxel_tensor = voxel_tensor * 2. - 1.
    return rescaled_voxel_tensor

def open_pickle(pickle_file):
    """Open a pickle file and return its contents.
    """
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data



def print_sentences(json_path, data_list):
    # Opens the processed captions generated from tools/preprocess_captions.py
    inputs_list = json.load(open(json_path, 'r'))
    idx_to_word = inputs_list['idx_to_word']

    if isinstance(data_list, list):
        sentences = convert_idx_to_words(idx_to_word, data_list)
    elif isinstance(data_list, np.ndarray):
        sentences = []
        for idx in range(data_list.shape[0]):
            sentences.append(('%04d  ' % idx) + ' '.join([idx_to_word[str(word_idx)]
                             for word_idx in data_list[idx, :] if word_idx != 0]))

    for sentence in sentences:
        print(sentence + '\n')





def categorylist2labellist(category_list_batch, category2label, opts): 
    """
    for primitive datasets, a batch data with category list:
        ['torus-cyan-h100-r20', 'torus-cyan-h100-r20', 'pyramid-orange-h50-r50', 
        'pyramid-orange-h50-r50', 'pyramid-yellow-h50-r100', 'pyramid-yellow-h50-r100', 
        'cone-purple-h50-r50', 'cone-purple-h50-r50']
    We convert it to be: 

    """
    if opts.dataset == 'shapenet':
        shape_labels = [category2label[cat] for cat in category_list_batch]
        if len(shape_labels) > opts.batch_size: # TST, MM 
            shape_label_batch = np.asarray(shape_labels[::opts.LBA_n_captions_per_model])
        else: # STS mode, validation 
            shape_label_batch = np.asarray(shape_labels) 
        return torch.from_numpy(shape_label_batch)
    elif opts.dataset == 'primitives':
        shape_labels = [category2label[cat] for cat in category_list_batch
                            for _ in range(opts.LBA_n_primitive_shapes_per_category)]

        if opts.LBA_model_type == 'TST' or opts.LBA_model_type == 'MM':
            shape_label_batch = np.asarray(shape_labels[::opts.LBA_n_captions_per_model])
        elif opts.LBA_model_type == 'STS': # STS mode, validation 
            # test_queue??? is false
            if opts.test_or_val_phase: # test or val phase 
                shape_label_batch = np.asarray(shape_labels)[::opts.LBA_n_primitive_shapes_per_category]
            else:  # we are in training phase 
                shape_label_batch = np.asarray(shape_labels)

        return torch.from_numpy(shape_label_batch)
    else: 
        raise ValueError('Please select a vlid dataset.')


