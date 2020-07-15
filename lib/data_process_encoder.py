import numpy as np 
import pickle 
import random 
import torch 
from collections import Counter 


from lib.utils import load_voxel
from multiprocessing import Process, Event 

class LBADataProcess(Process):#DataProcess
    """
    Data process that returns a raw caption batch and a shape batch 
    """
    def __init__(self, data_queue, data_dict, opts, repeat=True): #data_dict in input_dict
        ##################################################################
        ## data_dict: 
        ## keys: 
        ##   'caption_tuples': caption_tuples is
        ##        a list of caption tuples, where each caption tuple is (caption, model_category,
        ##        model_id). e.g., inputs_dict['caption_tuples'] = (array([1, 2, 3, 4, 5, 6, 0, 0], 
        ##                                  dtype=int32), 'cone-purple-h20-r100', 'cone-purple-h20-r100_7.nrrd')
        ##   'caption_matches': a dict where the key is any model ID and the value
        ##   is a list of the indices (ints) of caption tuples that describe the same model ID
        ##   'vocab_size': 
        ##   'max_caption_length': 
        ##   .......
        ##################################################################
        self.caption_matches = data_dict['caption_matches']

        self.data_queue = data_queue
        self.data_paths = range(len(self.caption_matches))
        self.num_data = len(self.data_paths) 
        self.repeat = repeat 


        self.opts = opts
        self.caption_tuples = data_dict['caption_tuples']

        
        
        

        self.batch_size = opts.batch_size 
        self.exit = Event() 

        self.matches_keys = list(self.caption_matches.keys())
        self.n_captions_per_model = opts.LBA_n_captions_per_model 

        
        self.n_unique_shape_categories = opts.batch_size 
        self.n_models_per_batch = self.n_unique_shape_categories
        
        super(LBADataProcess, self).__init__() 
        #super(LBADataProcess, self).__init__(data_queue, range(len(self.caption_matches)),batch_size=self.n_unique_shape_categories, repeat=repeat)

        self.max_sentence_length = len(self.caption_tuples[0][0])

        lengths = []
        for cur_tup in self.caption_matches.values():
            lengths.append(len(cur_tup))

        counter = Counter(lengths) 
        

        #load bad instances and neglect them
        with open(opts.probablematic_nrrd_path, 'rb') as f: 
            self.bad_model_ids = pickle.load(f) 

        self.shuffle_db_inds()
        self.iters_per_epoch = self.num_data // self.batch_size
        
    def shuffle_db_inds(self): 
        # Randomly permute the training roidb 
        if self.repeat: # if repeat is set to be True 
            self.perm = np.random.permutation(np.arange(self.num_data)) 
        else:
            self.perm = np.arange(self.num_data)

        # every time we shuffle the data, that means we are in the start of an epoch 
        # thus we set self.cur to be 0 
        self.cur = 0 

    def get_next_minibatch(self): 
        if (self.cur + self.batch_size) >= self.num_data and self.repeat: 
            # we exceed the dataset and self.repeat is True
            # then we shuffle the indexs(which will set self.cur to be 0) 
            self.shuffle_db_inds()
        # take out data 
        db_inds = self.perm[self.cur:min(self.cur+self.batch_size, self.num_data)]
        
        # update self.cur 
        self.cur += self.batch_size
        return db_inds # just return index of examples in the minibatch 

    # this will shut down this data process 
    def is_bad_model_id(self, model_id):
        if self.bad_model_ids is not None: 
            return model_id in self.bad_model_ids 
        else: 
            return False 

    def shutdown(self): 
        self.exit.set()  

    def verify_batch(self, caption_tuples_for_cur_key): 
        """
        simply verify that all caption tuples in the batch correspond to the same category and model id 
        """
        category = caption_tuples_for_cur_key[0][1]
        model_id = caption_tuples_for_cur_key[0][2] 
        for tup in caption_tuples_for_cur_key: 
            assert tup[1] == category
            assert tup[2] == model_id

        return category, model_id

    def run(self):

        """
        category and model lists dynamically change size depending on whether it is STS or TST mode 
        """
        # run the loop until exit flag is set 
        while not self.exit.is_set() and self.cur < self.num_data: 
            # print('{0}/{1} samples'.format(self.cur, self.num_data))

            # Ensure that the network sees (almost) all the data per epoch 
            db_inds = self.get_next_minibatch() 
            #print("1:",len(db_inds))
            shapes_list = []
            captions_list = []
            category_list = []
            model_id_list = []

            for db_ind in db_inds: # Loop through each selected shape 
                #print(db_ind)
                selected_shapes = [] 
                while True: 
                    # cur_key is the model id for shapenet, category for primitives 
                    #the self.matches_key list contains alls caption matches keys. Each key in caption matches is a model id
                    cur_key = self.matches_keys[db_ind] 
                    #each entry in caption_matches contains a few indices contraing the matches of captions to each other
                    caption_idxs = self.caption_matches[cur_key]
                    #print("2:",cur_key , len(caption_idxs))
                    ## Ensure theat len(caption_idxs) >= self.n_captions_per_model
                    if len(caption_idxs) < self.n_captions_per_model: # until len(caption_idxs) == self.n_captions_per_model
                        db_ind = np.random.randint(self.num_data) # take a random index
                        continue 

                    # randomly sample self.n_captions_per_model captions from caption_idxs
                    selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
                    selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs] 
                    #print("3:",len(selected_tuples))
                    # model id is cur_key  
                    cur_category, cur_model_id = self.verify_batch(selected_tuples) 
                    #print(cur_model_id)
                    # select shapes/models 
                    
                    selected_model_ids = [cur_model_id]

                    


                    # append cur_shape to selected_shapes  
                    # for shapenet, selected_model_ids = [cur_model_id]
                    # for primitives, category_model_ids = self.category2modelid[cur_category], and 
                    # we will saample self.LBA_n_primitive_shapes_per_category models for this category
                    for cur_model_id in selected_model_ids: 
                        if self.is_bad_model_id(cur_model_id):
                            db_ind = np.random.randint(self.num_data)
                            continue 
                        try: 
                            cur_shape = load_voxel(cur_category, cur_model_id, self.opts)

                        except FileNotFoundError: 
                            print('Error: cannot find file with the following model id: ', cur_key)
                            db_ind = np.random.randint(self.num_data) 
                            continue 
                        selected_shapes.append(cur_shape)
                    break
                # 每个model有self.n_captions_per_model个captions
                selected_captions = [tup[0] for tup in selected_tuples] 
                captions_list.extend(selected_captions)
                # 每个类（对于shapenet，选择1个），选择LBA_n_primitive_shapes_per_category个model
                for selected_shape in selected_shapes:  
                    shapes_list.append(selected_shape) 

               
                cur_categories = [cur_category for _ in selected_captions] 
                cur_model_ids = [cur_model_id for _ in selected_captions] 
                category_list.extend(cur_categories)
                model_id_list.extend(cur_model_ids) 
               

            # Length is the number of captions 
            # Index/label indicates which captions comes from the same shape 
            # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
            #label_list = [x for x in range(self.n_unique_shape_categories)
            #                for _ in range(self.n_captions_per_model)] 

            batch_captions = np.array(captions_list).astype(np.int32)
            batch_shapes = np.array(shapes_list).astype(np.float32) 
            # convert dim 
            #batch_shapes = batch_shapes.transpose((0, 4, 2,3,1)) # bz x 32 x 32 x 32 x 4 -> bz x 4 x 32 x 32 x 32
            #batch_label = np.array(label_list).astype(np.int32)  
             
            # item in the batch_data is pytorch Tensor 
            # the following will wait until the queue frees 
            
            batch_data = {
                "raw_embedding_batch": batch_captions, 
                'voxel_tensor_batch': batch_shapes, 
                #'caption_label_batch': batch_label, 
                'category_list':category_list, 
                'model_list':model_id_list, 
            }
            
            # kill_processes will run okay when the item in the batch_data is not tensor
            # batch_data = {
            #    "raw_embedding_batch": batch_captions.numpy(), 
            #    'voxel_tensor_batch': batch_shapes.numpy(), 
            #    'caption_label_batch': batch_label.numpy(), 
            #    'category_list':category_list, 
            #    'model_list':model_id_list, 
            #}

            self.data_queue.put(batch_data, block=True)

