import os
import torch

import argparse
import numpy as np
import random 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import lib.utils as ut
from lib.utils import load_voxel
import sys
sys.path.append("..")
from config import cfg
import pickle

def collate_wrapper_val(batch):
    
    captions = []
    shapes = []
    model_ids = []
    categories = []
    labels = []

    for i,dp in enumerate(batch):
        for k in dp[0]:
            #print(k.shape)
            captions.append(k)
            labels.append(i)
            model_ids.append(dp[2])
            categories.append(dp[3])

        shapes.append(dp[1])
        
   
    return torch.tensor(captions),torch.tensor(shapes),np.array(model_ids),np.array(categories),torch.tensor(labels)

def collate_wrapper_train(batch):
    
    captions = []
    shapes = []
    model_ids = []
    categories = []
    labels = []
    for i,dp in enumerate(batch):
        captions.append(dp[0])
        captions.append(dp[1])
        labels.append(i)
        labels.append(i)
        shapes.append(dp[2])
        model_ids.append(dp[3])
        model_ids.append(dp[3])
        categories.append(dp[4])

   
    return torch.tensor(captions),torch.tensor(shapes),np.array(model_ids),np.array(categories),torch.tensor(labels)

class ShapeNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, cap_per_model=2, opts=None, problematic_ones=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.opts = opts
        self.caption_matches = data_dict['caption_matches']
        self.matches_keys = list(self.caption_matches.keys())
        self.num_data = len(self.caption_matches) 

        self.caption_tuples = data_dict['caption_tuples']
        
        self.n_captions_per_model = cap_per_model



        self.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
        with open(self.probablematic_nrrd_path, 'rb') as f: 
            self.bad_model_ids = pickle.load(f)

    

    def __len__(self):
        return self.num_data

    

    def __getitem__(self, idx):


        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        db_ind = idx
        while(True):
            cur_key = self.matches_keys[db_ind]
            caption_idxs = self.caption_matches[cur_key]
            if len(caption_idxs) < self.n_captions_per_model: # until len(caption_idxs) == self.n_captions_per_model
                db_ind = np.random.randint(self.num_data) # take a random index
                continue
            

            selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
            selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs] 
            cur_model_id = selected_tuples[0][2]
            cur_category = selected_tuples[0][1]

            #if(cur_category == '3d734hd'):
            #    db_ind = np.random.randint(self.num_data)
            #    continue

            selected_model_ids = cur_model_id

            if cur_model_id in self.bad_model_ids:
                db_ind = np.random.randint(self.num_data)
                continue
            else:
                break 
        cur_shape = load_voxel(cur_category, cur_model_id, self.opts)

        return selected_tuples[0][0],selected_tuples[1][0], cur_shape, cur_model_id, cur_category

       
    
class ShapeNetDataset_Validation(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_dict, cap_per_model=4,opts=None,problematic_ones=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.opts = opts
        self.caption_matches = data_dict['caption_matches']
        self.matches_keys = list(self.caption_matches.keys())
        self.num_data = len(self.caption_matches) 

        self.caption_tuples = data_dict['caption_tuples']
        
        self.n_captions_per_model = cap_per_model



        self.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
        with open(self.probablematic_nrrd_path, 'rb') as f: 
            self.bad_model_ids = pickle.load(f)

    

    def __len__(self):
        return self.num_data

    

    def __getitem__(self, idx):


        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        db_ind = idx
        while(True):
            cur_key = self.matches_keys[db_ind]
            caption_idxs = self.caption_matches[cur_key]
            if len(caption_idxs) < self.n_captions_per_model: # until len(caption_idxs) == self.n_captions_per_model
                db_ind = np.random.randint(self.num_data) # take a random index
                continue
            

            selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
            selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs] 
            cur_model_id = selected_tuples[0][2]
            cur_category = selected_tuples[0][1]
            selected_model_ids = cur_model_id

            if cur_model_id in self.bad_model_ids:
                db_ind = np.random.randint(self.num_data)
                continue
            else:
                break 
        cur_shape = load_voxel(cur_category, cur_model_id, self.opts)

        selected_tuples = [s[0] for s in selected_tuples]
        return selected_tuples, cur_shape, cur_model_id, cur_category



#Test ---------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main text2voxel train/test file')
    opts = parser.parse_args()
    opts.data_dir = cfg.DIR.RGB_VOXEL_PATH

    inputs_dict = ut.open_pickle(cfg.DIR.TRAIN_DATA_PATH)
    params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6,
           'captions_per_model':2 }

    params_2 = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6,
          'collate_fn' : collate_wrapper
           }
    dat_loader = ShapeNetDataset(inputs_dict,params,opts)
    
    training_generator = torch.utils.data.DataLoader(dat_loader, **params_2)

    #mask_ndarray = np.asarray([1., 0.] * (64))[:, np.newaxis]
    #mask = torch.from_numpy(mask_ndarray).float().type_as(text_embeddings.data).expand(text_embeddings.size(0), text_embeddings.size(1))
    #inverted_mask = 1. - mask
    #embeddings = text_embeddings * mask + shape_embeddings_rep * inverted_mask
    print("LENGTH",len(training_generator))

    for captions,shapes,model_ids,categories,labels in training_generator:
        
        print(captions.shape)
        
        #print(captions, shapes, model_ids, categories)
        