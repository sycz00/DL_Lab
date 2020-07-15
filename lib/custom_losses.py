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
        
        #self.trip_loss = TripletLoss(margin = 0.5)

        
    def mahalanobis(self,x, y, cov=None):
        x_mean = torch.mean(x)
        Covariance = self.cov(y)#self.cov(torch.transpose(y,0,1))
        inv_covmat = torch.inverse(Covariance)
        x_minus_mn = x - x_mean
        
        
        D_square = torch.mm(torch.mm(x_minus_mn, inv_covmat), torch.transpose(x_minus_mn,0,1))
        return D_square 
  

    def cov(self,m, rowvar=False):
    
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()  # if complex: mt = m.t().conj()
        return fact * m.matmul(mt).squeeze()


    
    
    def smoothed_metric_loss(self, input_tensor, margin=1,similarity='dot'): 
        """
         Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        input_tensor: size: N x emb_size 
        """ 
       

        X = input_tensor # N x emb_size 
        m = margin 

        if(similarity != 'dot'):
            #mahanaobis distance instead of simple dot product
            D = self.mahalanobis(X,X)
            expmD = torch.exp(m - D)
            #magnitude = (input_tensor ** 2).sum(1).expand(self.batch_size, self.batch_size)
            #squared_matrix = input_tensor.mm(torch.t(input_tensor))
            #D = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()#mahalanobis_distances
            #expmD = torch.exp(m - D)
        else:
            D = torch.mm(X,X.transpose(0, 1))
            #D /= 128 #if not normalized in encoder
            expmD = torch.exp(m + D)



        
        J_all = []#Variable(torch.zeros(1), requires_grad=True)
        #counter = 0 

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
            if(similarity != 'dot'):
                J_ij = torch.square(F.relu(torch.log(torch.sum(expmD[neg_inds])) + D[i, j]))
            else:
                J_ij = torch.square(F.relu(torch.log(torch.sum(expmD[neg_inds])) - D[i, j]))
                
            #J_ij = torch.log(torch.sum(expmD[neg_inds])) + D[i, j]
            J_all.append(J_ij) #torch.square(F.relu(J_ij)) 
            #counter += 1 


        
        J_all = torch.stack(J_all)
        
        #loss = torch.mean(F.relu(J_all)**2)*0.5 #mean represents |P| and therefore only 1/2 remains to be multiplied with 
        loss = torch.mean(J_all)*0.5#J_all/(2*counter)
        
        return loss 

    



    def forward(self, text_embeddings, shape_embeddings):
        
       # we may rewrite batch_size 
        self.batch_size = text_embeddings.size(0)
        
        indices = [i // 2 for i in range(shape_embeddings.size(0) * self.LBA_n_captions_per_model)]
        shape_embeddings_rep = torch.index_select(shape_embeddings, 0, torch.LongTensor(indices).cuda())
        

        embeddings = text_embeddings  
        #T-T LOSS
        metric_tt_loss= self.smoothed_metric_loss(embeddings, self.cur_margin) 
        
        
        

        mask_ndarray = np.asarray([1., 0.] * (self.batch_size//2))[:, np.newaxis] #[0,1,0,1,0,1] shape[200,1]
        
        
        mask = torch.from_numpy(mask_ndarray).float().type_as(text_embeddings.data).expand(text_embeddings.size(0), text_embeddings.size(1)) #converts it to [200,128]
        
        inverted_mask = 1. - mask
        
        # (text_1_emb, shape_emb_1, ..., text_N_emb, shape_emb_N) (the consective two are the same label)    
        embeddings = text_embeddings * mask + shape_embeddings_rep * inverted_mask
        # (text_2_emb,shape_emb_1,....)  
        #embeddings_2 = text_embeddings * inverted_mask + shape_embeddings_rep * mask
        #embeddings = torch.cat([embeddings_1,embeddings_2],axis=0)
        
       
        self.batch_size = embeddings.size(0)
        #T-S Loss
        metric_st_loss = self.smoothed_metric_loss(embeddings,self.cur_margin)
        
        
        

        Total_loss = metric_tt_loss +  metric_st_loss
        #Total_loss_with_norm = Total_loss + self.text_norm_weight * unweighted_txt_loss + self.shape_norm_weight * unweighted_shape_loss

        
       
        
       
        return Total_loss 

"""
def euclidean_distance(self, X, Y):
        p1 = torch.sum(X**2,axis=1).unsqueeze(1)
        p2 = torch.sum(Y**2,axis=1)
        p3 = -2* torch.mm(X,Y.transpose(0,1))
        return torch.sqrt(p1+p2+p3+1e-8)
        #m, p = X.size() 
        #n, p = Y.size() 
        #X_exp = torch.stack([X]*n).transpose(0,1)
        #Y_exp = torch.stack([Y]*m)
        #dist = torch.sum((X_exp-Y_exp)**2,2).squeeze() # size: m x n 
        #dist = (dist+1e-8).sqrt_() # applies inplace sqrt 
        #return dist
"""
#Regulization via norm of weights... (perhaps not needed in the case of metric learning)
#text_norms = torch.norm(text_embeddings, p=2, dim=1)
#unweighted_txt_loss = torch.mean(F.relu(text_norms - self.LBA_max_norm))
#shape_norms = torch.norm(shape_embeddings_rep, p=2, dim=1)
#unweighted_shape_loss = torch.mean(F.relu(shape_norms - self.LBA_max_norm))