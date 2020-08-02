import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable




def pairwise_dot(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: sim_mat is a NxM matrix where sim_mat[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. sim_mat[i,j] = x[i,:] * y[j,:]
    '''
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
    else:
        y_t = torch.transpose(x, 0, 1)
    
    sim_mat = torch.mm(x, y_t)

    return sim_mat


class softCrossEntropy_v2(nn.Module):
    def __init__(self):
        super(softCrossEntropy_v2, self).__init__()

    def forward(self, inputs, soft_target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        # take log function 
        inputs = torch.log(inputs + 1e-8)
        inputs = F.softmax(inputs, dim=1)
        cross_entropy_loss = soft_target * torch.log(inputs) # + (1 - soft_target) * torch.log(1-inputs) 
        cross_entropy_loss = -1 * cross_entropy_loss 
        loss = torch.mean(torch.sum(cross_entropy_loss, dim=1))

        return loss

class Semisup_Loss(nn.Module):
    def __init__(self, lmbda=1, lba_dist_type='standard'):
        """
        semi-supervised classification loss to the model, 
        The loss constist of two terms: "walker" and "visit".
        Args:
        A: [N, emb_size] tensor with supervised embedding vectors.
        B: [M, emb_size] tensor with unsupervised embedding vectors.
        labels : [N] tensor with labels for supervised embeddings.
        walker_weight: Weight coefficient of the "walker" loss.
        visit_weight: Weight coefficient of the "visit" loss.
        Return: 
            return the sum of two losses 
        """
        super(Semisup_Loss, self).__init__() 
        self.lmbda = lmbda 
        self.lba_dist_type = lba_dist_type

        # note nn.CrossEntropyLoss() only support the case when target is categorical value, e.g., 0, 1, ..
        self.cross_entropy = softCrossEntropy_v2() #

        self.mse_loss = nn.MSELoss()


    def forward(self, A, B, labels): 
        """
        compute similarity matrix 
        Args: 
            A: size: N x emb_size, tensor with supervised embedding vectors 
            B: size: M x emb_size, tensor with unsupervised embedding vectors  
            labels: size: N, tensor with labels for supervised embeddings 
            for ease of understanding, currently, A -> text_embedding, B -> shape_embedding, labels -> caption_labels
        """ 
        # build target probability distribution matrix based on uniform dist over correct labels 
        # N x N 
        """
        a1 = [[ 0,  0,  1,  ..., 98, 99, 99],
        [ 0,  0,  1,  ..., 98, 99, 99],
        [ 0,  0,  1,  ..., 98, 99, 99],
        ...,
        [ 0,  0,  1,  ..., 98, 99, 99],
        [ 0,  0,  1,  ..., 98, 99, 99],
        [ 0,  0,  1,  ..., 98, 99, 99]]

        a2 = [[ 0,  0,  0,  ...,  0,  0,  0],
        [ 0,  0,  0,  ...,  0,  0,  0],
        [ 1,  1,  1,  ...,  1,  1,  1],
        ...,
        [98, 98, 98,  ..., 98, 98, 98],
        [99, 99, 99,  ..., 99, 99, 99],
        [99, 99, 99,  ..., 99, 99, 99]]
        """

        """ equality matrix :
        [[ True,  True, False,  ..., False, False, False],
        [ True,  True, False,  ..., False, False, False],
        [False, False,  True,  ..., False, False, False],
        ...,
        [False, False, False,  ...,  True, False, False],
        [False, False, False,  ..., False,  True,  True],
        [False, False, False,  ..., False,  True,  True]]

        the sum is just [2,2,2,2,.....,2]

        p_target =
        [[0.5000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.5000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.5000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.5000, 0.5000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.5000, 0.5000]]
        """


        equality_matrix = torch.eq(*[labels.unsqueeze(dim).expand(A.size(0), A.size(0)) for dim in [0, 1]]).type_as(A)
        p_target = torch.div(equality_matrix, torch.sum(equality_matrix, dim=1, keepdim=True))

        
        M = pairwise_dot(A, B) # N x M, each row i: sim(text_i, shape_1), sim(text_i, shape_2),sim(text_i, shape_3) ...
        
            

        M_t = M.transpose(0, 1).contiguous() # M x N, each row i: sim(shape_i, text_1), sim(shape_i, text_2),sim(shape_i, text_3) ...
        
        P_TS_distr = F.softmax(M, dim=1) # N x M 
        P_ST_distr = F.softmax(M_t, dim=1)  # M x N 
        # text-shape-text round trip 
        #so the probability of starting in T_i go over all shapes and come back to different T_j the 
        P_TST = torch.mm(P_TS_distr, P_ST_distr) # N x N 
        
       
        
        ################################################
        # may be we should use mse instead of soft label cross entropy loss 
        ################################################
        # we will first take log function on P_TST and then applity softmax 
        L_TST_r = self.cross_entropy(P_TST, p_target)
        #L_TST_r = self.mse_loss(P_TS_distr)

        ################################################
        ## To associate text descriptions with all possible matching shapes
        ## we impose loss on the probability of asscociating each shape (j) with 
        ## any descriptions
        ################################################
        P_visit = torch.mean(P_TS_distr, dim=0, keepdim=True) # N text, M shape, (N x M) => 1 x M 
        
        soft_target2 = torch.ones(1, P_visit.size(1)).type_as(P_visit.data)/P_visit.size(1) # 1 x M
        L_TST_h = self.cross_entropy(P_visit, soft_target2)

        Total_loss = L_TST_r + self.lmbda * L_TST_h

        return Total_loss, P_TST, p_target

class LBA_Loss(nn.Module):

    def __init__(self, lmbda=1.0, batch_size=None):
        super(LBA_Loss, self).__init__() 
        
        self.lmbda = lmbda
        self.batch_size = batch_size 
        self.semisup_loss = Semisup_Loss(self.lmbda, lba_dist_type='standard')

    def forward(self, text_embedding, shape_embedding, labels): 
        """
        note that the returned P_STS and P_Target_TST e.t.c are nothing but for display purpose
        """
        # pdb.set_trace()
        # during test when we use this criterion, we may not get self.batch_size data 
        # so ..
        #self.batch_size = text_embedding.size(0)

        #labels for the captions : always two instances each class => uni(0.5,0.5)
        A = text_embedding
        self.batch_size = A.size(0)
        B = shape_embedding    
        TST_loss, _, _ = self.semisup_loss(A, B, labels) 


        #only one instance per class => predict the 1.0
        B = text_embedding
        A = shape_embedding
        self.batch_size = A.size(0)
        labels = torch.from_numpy(np.array(range(self.batch_size))).type_as(A.data)
        STS_loss, _, _ = self.semisup_loss(A, B, labels)

        return TST_loss, STS_loss,#P_TST, P_target_TST
        

################################################################### 
## Metric loss 
###################################################################


class Metric_Loss(nn.Module):
    """
    used only for training 
    """
    def __init__(self, opts):
        super(Metric_Loss, self).__init__() 
        

        assert opts.LBA_n_captions_per_model == 2 
        
        self.dataset = opts.dataset
        self.LBA_n_captions_per_model = opts.LBA_n_captions_per_model
        self.batch_size = opts.batch_size
        
        
        self.cur_margin = 0.5       
        #Penalize Text and shape embedding norms if encoder do not normalize 
        #self.LBA_max_norm = LBA_max_norm
        #self.text_norm_weight = 2.0 #as used in the official implementation
        #self.shape_norm_weight = 2.0 ##as used in the official implementation
        
        #self.trip_loss = TripletLoss(margin = 0.5)

    def euclidean_distance(self, X, Y):
        p1 = torch.sum(X**2,axis=1).unsqueeze(1)
        p2 = torch.sum(Y**2,axis=1)
        p3 = -2* torch.mm(X,Y.transpose(0,1))
        return torch.sqrt(p1+p2+p3+1e-8)
        


    def pairwise_distances(self,x, y=None):
        x_norm = (x**2).sum(1).view(-1, 1)

        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return dist#torch.clamp(dist, 0.0, np.inf)

        
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


    
    


    def smoothed_metric_loss(self, input_tensor, margin=0.5,similarity='dot'): 
        """
         Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
        input_tensor: size: N x emb_size 
        """ 
       
        

        X = input_tensor # N x emb_size 
        m = self.cur_margin

            
        #X = F.normalize(X, p=2, dim=1)
        #D = self.pairwise_distances(X,X)#self.euclidean_distance(X,X)
        D = torch.mm(X,X.transpose(0, 1))##nn.CosineSimilarity(dim=1, eps=1e-6)#
        
        #D /= 128 #if not normalized in encoder
        #D = 1.0 - D
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
            #if(similarity != 'dot'):
            #    J_ij = torch.square(F.relu(torch.log(torch.sum(expmD[neg_inds])) + D[i, j]))
            #else:
            J_ij = torch.square(F.relu(torch.log(torch.sum(expmD[neg_inds])) - D[i, j]))
            #J_ij = torch.log(torch.sum(expmD[neg_inds])) - D[i, j]
                
            #J_ij = torch.log(torch.sum(expmD[neg_inds])) + D[i, j]
            J_all.append(J_ij) #torch.square(F.relu(J_ij)) 
            #counter += 1 


        
        J_all = torch.stack(J_all)
        
        loss = torch.div(torch.mean(J_all),2)#*0.5 #mean represents |P| and therefore only 1/2 remains to be multiplied with 
        #loss = torch.mean(J_all)*0.5#J_all/(2*counter)
        
        return loss 

    



    def forward(self, text_embeddings, shape_embeddings):
        
       # we may rewrite batch_size 
        self.batch_size = text_embeddings.size(0)
        
        indices = [i // 2 for i in range(shape_embeddings.size(0) * self.LBA_n_captions_per_model)]
        shape_embeddings_rep = torch.index_select(shape_embeddings, 0, torch.LongTensor(indices).cuda())
        

        embeddings = text_embeddings  
        #T-T LOSS
        metric_tt_loss= self.smoothed_metric_loss(embeddings, self.cur_margin) #self.triplet_loss(embeddings)#
        
        
        

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
        metric_st_loss = self.smoothed_metric_loss(embeddings,self.cur_margin)#self.triplet_loss(embeddings)#
        
        #metric_ss_loss = self.smoothed_metric_loss(shape_embeddings,self.cur_margin)
        

        Total_loss = metric_tt_loss + 2.* metric_st_loss
        #Total_loss_with_norm = Total_loss + self.text_norm_weight * unweighted_txt_loss + self.shape_norm_weight * unweighted_shape_loss

        
        #text_norms = torch.norm(text_embeddings, p=2, dim=1)
        #unweighted_txt_loss = torch.mean(F.relu(text_norms - self.LBA_max_norm))
        #shape_norms = torch.norm(shape_embeddings_rep, p=2, dim=1)
        #unweighted_shape_loss = torch.mean(F.relu(shape_norms - self.LBA_max_norm))
        #Total_loss_with_norm = Total_loss + self.text_norm_weight * unweighted_txt_loss + self.shape_norm_weight * unweighted_shape_loss
        
       
        return Total_loss #Total_loss_with_norm#

"""

"""
#Regulization via norm of weights... (perhaps not needed in the case of metric learning)
"""
def triplet_loss(self,input_tensor, name='triplet_loss', margin=1.):
    
        
            # Song et al., Deep Metric Learning via Lifted Structured Feature Embedding
            # Define feature X 
        X = input_tensor
        m = margin

        # Compute the pairwise distance
        Xe = X.unsqueeze(1)#tf.expand_dims(X, 1)
        
        
        Dsq = torch.sum(torch.square(Xe - Xe.permute(1, 0, 2)),2)#tf.reduce_sum(tf.square(Xe - tf.transpose(Xe, (1, 0, 2))), 2)
        D = torch.sqrt(Dsq + 1e-8)
        mD = m - D

        # Compute the loss
        # Assume that the input data is aligned in a way that two consecutive data form a pair
        batch_size= X.size()[0]#X.get_shape().as_list()

        # L_{ij} = 
        for pair_ind in range(batch_size // 2):
            i = pair_ind * 2
            j = i + 1
            ind_rest = np.hstack([np.arange(0, pair_ind * 2),
                                  np.arange(pair_ind * 2 + 2, batch_size)])

            inds = [[i, k] for k in ind_rest]
            inds.extend([[j, l] for l in ind_rest])

            J_ij = torch.max(mD[inds]+D[[i,j]])#tf.reduce_max(tf.gather_nd(mD, inds)) + tf.gather_nd(D, [[i, j]])
            J_all.append(J_ij)

        J_all = torch.stack(J_all)#tf.convert_to_tensor(J_all)
        #loss = tf.divide(tf.reduce_mean(tf.square(tf.maximum(J_all, 0))), 2.0, name='metric_loss')
        
        
         
        #J_max,_ = torch.max(J_all,0)
        loss = torch.div(torch.mean(torch.square(J_all)),2.0)
        #tf.losses.add_loss(loss)
        return loss
"""