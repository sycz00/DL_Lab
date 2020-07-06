import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable





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
        #if LBA_inverted_loss is True: 
        #self.cur_margin = 1.0 
        #else: 
        self.cur_margin = 1.0 

        ################################################
        ## should we specify the self.text_norm_weight and self.shape_norm_weight 
        ## here we add a penalty on the embedding norms  
        ################################################
        
        self.LBA_max_norm = LBA_max_norm
        self.text_norm_weight = 2.0 #as used in the official implementation
        self.shape_norm_weight = 2.0 ##as used in the official implementation
        


        
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
        #implemented by paper formular :
        #Deep Metric Learning via Lifted Structured Feature Embedding

        X = input_tensor # N x emb_size 
        m = margin 

        #mahanaobis distance instead of simple dot product
        magnitude = (input_tensor ** 2).sum(1).expand(self.batch_size, self.batch_size)
        squared_matrix = input_tensor.mm(torch.t(input_tensor))
        D = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()#mahalanobis_distances
        #if self.LBA_cosin_dist is True: 
            #assert (self.LBA_normalized is True) or (self.LBA_inverted_loss is True) 
            #assert (self.LBA_normalized is True) and (margin < 1) or (self.LBA_inverted_loss is True)

            #D = self.cosine_similarity(X, X) #the m_i_j in the equation 2
           
        expmD = torch.exp(m - D)

        # compute the loss 
        # assume that the input data is aligned in a way that two consective data form a pair 

        
        J_all = 0#Variable(torch.zeros(1), requires_grad=True)
        counter = 0 
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
            #dirty implemented but works
            neg_row_ids = [int(coord[0]) for coord in neg_inds]
            neg_col_ids = [int(coord[1]) for coord in neg_inds]

            neg_inds = [neg_row_ids, neg_col_ids]

            J_ij = torch.log(torch.sum(expmD[neg_inds])) + D[i, j]

            J_all += torch.square(F.relu(J_ij)) 
            counter += 1 


        #P_len = len(J_all)
        #J_all = torch.stack(J_all)
        
        #loss = torch.mean((F.relu(J_all)**2))*0.5 #mean represents |P| and therefore only 1/2 remains to be multiplied with 
        loss = J_all/(2*counter)
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
        #targets = Variable(torch.IntTensor([i // 2 for i in range(embeddings.size(0))])).cuda()
        
        

        mask_ndarray = np.asarray([1., 0.] * (self.batch_size//2))[:, np.newaxis] #[0,1,0,1,0,1] shape[200,1]
        
        
        mask = torch.from_numpy(mask_ndarray).float().type_as(text_embeddings.data).expand(text_embeddings.size(0), text_embeddings.size(1)) #converts it to [200,128]
        
        inverted_mask = 1. - mask
        
        # text_1_emb, shape_emb_1, ..., text_N_emb, shape_emb_N (the consective two are the same label)
        # therefore we neglect the half of the captions. 
        # thus using again the second text embedding to include both captions per shape.       
        embeddings_1 = text_embeddings * mask + shape_embeddings_rep * inverted_mask
        embeddings_2 = text_embeddings * inverted_mask + shape_embeddings_rep * mask
        embeddings = torch.cat([embeddings_1,embeddings_2],axis=0)
        
       
        self.batch_size = embeddings.size(0)
        metric_st_loss = self.smoothed_metric_loss(embeddings,self.cur_margin)
        #targets = Variable(torch.IntTensor([i // 2 for i in range(embeddings.size(0))])).cuda()
        #loss_ST, _, _, _ = self.test_loss(embeddings,targets)
        #loss_ST = self.lifted_loss(embeddings)
        
        
        
        
        Total_loss = metric_tt_loss +  metric_st_loss

       
        return Total_loss 

 
