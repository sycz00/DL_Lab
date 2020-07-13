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

    def build_walk_stastistics(self, p_tst, equality_matrix):
        """
        compute walker loss stastistics
        Args: 
            p_tst: N x N matrix, where [j,j] element corresponds to the probability 
            of the round-trip between supervised samples i and j 
            sum of each row of p_tst must be equal to 1
            equality_matrix: N x N matrix, [i,j] == 1 -> samples i and j belong to the same class   
        """
        # Using the sequare root of the correct round trip probability as an estimate of the 
        # current classifier accuracy 
        per_row_accuracy = 1.0 - torch.sum(equality_matrix * p_tst, dim=1) ** 0.5 
        estimate_error = torch.mean(1-per_row_accuracy)
        return estimate_error


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
        equality_matrix = torch.eq(*[labels.unsqueeze(dim).expand(A.size(0), A.size(0)) for dim in [0, 1]]).type_as(A)
        p_target = torch.div(equality_matrix, torch.sum(equality_matrix, dim=1, keepdim=True))

        if self.lba_dist_type == 'standard':
            M = pairwise_dot(A, B) # N x M, each row i: sim(text_i, shape_1), sim(text_i, shape_2),sim(text_i, shape_3) ...
        else: 
            return ValueError('please select a valid distance type')

        M_t = M.transpose(0, 1).contiguous() # M x N, each row i: sim(shape_i, text_1), sim(shape_i, text_2),sim(shape_i, text_3) ...
        
        P_TS_distr = F.softmax(M, dim=1) # N x M 
        P_ST_distr = F.softmax(M_t, dim=1)  # M x N 
        # text-shape-text round trip 
        P_TST = torch.mm(P_TS_distr, P_ST_distr) # N x N 
        # build walk stastistics using equality matrix 
        estimate_error = self.build_walk_stastistics(P_TST, equality_matrix)
        
        ################################################
        # may be we should use mse instead of soft label cross entropy loss 
        ################################################
        # we will first take log function on P_TST and then applity softmax 
        L_TST_r = self.cross_entropy(P_TST, p_target)

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

    def __init__(self, lmbda=1.0, LBA_model_type='MM', batch_size=None):
        super(LBA_Loss, self).__init__() 
        self.LBA_model_type = LBA_model_type 
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
        self.batch_size = text_embedding.size(0)

        if self.LBA_model_type == 'MM' or self.LBA_model_type == 'TST': 
            A = text_embedding
            B = shape_embedding 
            # TST_loss = L^{TST}_R + \lambda * L^{TST}_H, see equation (1) in the text2shape paper 
            TST_loss, P_TST, P_target_TST = self.semisup_loss(A, B, labels) 
        if self.LBA_model_type == 'MM' or self.LBA_model_type == 'STS':
            B = text_embedding
            A = shape_embedding
            labels = torch.from_numpy(np.array(range(self.batch_size))).type_as(A.data)
            # see equation (3) in the paper 
            # STS_loss = L^{TST}
            STS_loss, P_STS, P_target_STS = self.semisup_loss(A, B, labels)
    

        if self.LBA_model_type == 'MM':
            # see equaiton (3) in the text2shape paper 
        
            return TST_loss + STS_loss, P_TST, P_target_TST

        if self.LBA_model_type == 'TST':
            return TST_loss, P_TST, P_target_TST

        if self.LBA_model_type == 'STS': 
            return STS_loss, P_STS, P_target_STS


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
        self.cur_margin = 0.5 

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
        #magnitude = (input_tensor ** 2).sum(1).expand(self.batch_size, self.batch_size)
        #squared_matrix = input_tensor.mm(torch.t(input_tensor))
        #D = F.relu(magnitude + torch.t(magnitude) - 2 * squared_matrix).sqrt()#mahalanobis_distances
        #if self.LBA_cosin_dist is True: 
            #assert (self.LBA_normalized is True) or (self.LBA_inverted_loss is True) 
            #assert (self.LBA_normalized is True) and (margin < 1) or (self.LBA_inverted_loss is True)

        D = self.cosine_similarity(X, X) #the m_i_j in the equation 2
           
        expmD = torch.exp(m + D)

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

            J_ij = torch.square(torch.log(torch.sum(expmD[neg_inds])) - D[i, j])

            J_all += J_ij #torch.square(F.relu(J_ij)) 
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

 
