import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 






################################################################### 
## Metric loss 
###################################################################


class Metric_Loss(nn.Module):
    """
    used only for training 
    """
    def __init__(self, opts, LBA_inverted_loss=True, LBA_normalized=True, LBA_max_norm=None):
        super(Metric_Loss, self).__init__() 
        # either is true 
        assert (LBA_inverted_loss is True) or (LBA_normalized is True)
        assert opts.LBA_n_captions_per_model == 2 
        self.LBA_inverted_loss = LBA_inverted_loss 
        self.LBA_normalized = LBA_normalized
        self.dataset = opts.dataset
        self.LBA_n_captions_per_model = opts.LBA_n_captions_per_model
        self.batch_size = opts.batch_size
        self.LBA_cosin_dist = opts.LBA_cosin_dist
        if self.dataset == 'primitives':
            self.LBA_n_primitive_shapes_per_category = opts.LBA_n_primitive_shapes_per_category
            assert self.LBA_n_primitive_shapes_per_category == 2 
        if LBA_inverted_loss is True: 
            self.cur_margin = 1. 
        else: 
            self.cur_margin = 0.5 

        ################################################
        ## should we specify the self.text_norm_weight and self.shape_norm_weight 
        ## here we add a penalty on the embedding norms  
        ################################################
        if LBA_max_norm is not None: 
            self.LBA_max_norm = LBA_max_norm
            self.text_norm_weight = 2.0 #as used in the official implementation
            self.shape_norm_weight = 2.0 ##as used in the official implementation
        else: # default value 
            self.LBA_max_norm = LBA_max_norm
            self.text_norm_weight = 2.0 
            self.shape_norm_weight = 2.0 


    #######################################################
    ##
    #######################################################
    def cosine_similarity(self, X, Y):

        
        Y_t = Y.transpose(0, 1)

        K = torch.mm(X, Y_t)

        return K
    

    

    #######################################################
    ##
    ########################################################
    def smoothed_metric_loss(self, input_tensor, margin=1): 
        """
         Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        input_tensor: size: N x emb_size 
        """ 
        # compute pairwise distance 
        X = input_tensor # N x emb_size 
        m = margin 

        if self.LBA_cosin_dist is True: 
            assert (self.LBA_normalized is True) or (self.LBA_inverted_loss is True) 
            assert (self.LBA_normalized is True) and (margin < 1) or (self.LBA_inverted_loss is True)

            D = self.cosine_similarity(X, X) #the m_i_j in the equation 2
           

            if self.LBA_inverted_loss is False: 
                D = 1.0 - D 
            else: 
                D /= 128. 
        else: 
            D = self.euclidean_distance(X, X)

        if self.LBA_inverted_loss is True:

            expmD = torch.exp(m + D)
            
        else: 
            expmD = torch.exp(m - D)

        # compute the loss 
        # assume that the input data is aligned in a way that two consective data form a pair 

        # L_{ij} = \log (\sum_{i, k} exp\{m - D_{ik}\} + \sum_{j, l} exp\{m - D_{jl}\}) + D_{ij}
        # L = \frac{1}{2|P|}\sum_{(i,j)\in P} \max(0, J_{i,j})^2
        J_all = [] 
        for pair_ind in range(self.batch_size//2): 
            i = pair_ind * 2 # 0, 2, 4, ...
            j = i + 1 # j is the postive of i 

            # the rest is the negative indices 
            # 0, ..., i-1, exclude(i, i+1),  i + 2, ..., self.batch_size
            
            #if i = 0 and j = 1 the negatives are all other in the batch except i and j
            #this can be done because they are always distinct in each batch
            ind_rest = np.hstack([np.arange(0, pair_ind * 2), np.arange(pair_ind * 2 + 2, self.batch_size)])
            

            
            neg_inds = [[i, k] for k in ind_rest]
            neg_inds.extend([[j, l] for l in ind_rest])

            neg_row_ids = [int(coord[0]) for coord in neg_inds]
            neg_col_ids = [int(coord[1]) for coord in neg_inds]
            neg_inds = [neg_row_ids, neg_col_ids]

            if self.LBA_inverted_loss is True: 
                J_ij = torch.log(torch.sum(expmD[neg_inds])) - D[i, j]
            else: 
                J_ij = torch.log(torch.sum(expmD[neg_inds])) + D[i, j]

            J_all.append(J_ij) 

        
        J_all = torch.stack(J_all)
        
        loss = torch.mean((J_all)**2) * 0.5 #removed F.relu inside mean
    
        return loss 



    def forward(self, text_embeddings, shape_embeddings):
        
       # we may rewrite batch_size 
        self.batch_size = text_embeddings.size(0)
        
        indices = [i // 2 for i in range(shape_embeddings.size(0) * self.LBA_n_captions_per_model)]
        shape_embeddings_rep = torch.index_select(shape_embeddings, 0, torch.LongTensor(indices).cuda())
        

        ##############################################################
        ## TT loss 
        ##############################################################
        embeddings = text_embeddings  
        metric_tt_loss= self.smoothed_metric_loss(embeddings, self.cur_margin) 

        mask_ndarray = np.asarray([1., 0.] * (self.batch_size//2))[:, np.newaxis] #[0,1,0,1,0,1] shape[200,1]
        
        
        mask = torch.from_numpy(mask_ndarray).float().type_as(text_embeddings.data).expand(text_embeddings.size(0), text_embeddings.size(1)) #converts it to [200,128]
        
        inverted_mask = 1. - mask
        
        # text_1_emb, shape_emb_1, ..., text_N_emb, shape_emb_N (the consective two are the same label)
        # therefore we neglect the half of the captions. 
        # thus using again the second text embedding to include both captions per shape.       
        embeddings = text_embeddings * mask + shape_embeddings_rep * inverted_mask
        #embeddings_2 = text_embeddings * inverted_mask + shape_embeddings_rep * mask
        #embeddings = torch.cat([embeddings_1,embeddings_2],axis=0)
        
       
        self.batch_size = embeddings.size(0)
        metric_st_loss = self.smoothed_metric_loss(embeddings,self.cur_margin)

        # embeddings = text_embeddings * inverted_mask + shape_embeddings_rep * mask
        # metric_ts_loss = self.smoothed_metric_loss(embeddings, name='smoothed_metric_loss_ts', margin=cur_margin)

        Total_loss = metric_tt_loss +  metric_st_loss


        if self.LBA_normalized is False:  # Add a penalty on the embedding norms
            """
            only when self.LBA_normalizd is False
            """
            text_norms = torch.norm(text_embeddings, p=2, dim=1)
            unweighted_txt_loss = torch.mean(F.relu(text_norms - self.LBA_max_norm))
            shape_norms = torch.norm(shape_embeddings, p=2, dim=1)
            unweighted_shape_loss = torch.mean(F.relu(shape_norms - self.LBA_max_norm))

            Total_loss_with_norm = Total_loss + self.text_norm_weight * unweighted_txt_loss + self.shape_norm_weight * unweighted_shape_loss
            
            return Total_loss_with_norm
        else: 
            return Total_loss 

 
