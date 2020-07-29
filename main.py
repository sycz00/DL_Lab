import argparse
import numpy as np
import os


import torch
import torch.optim as optim
# HACK: Get logger to print to stdout
import sys

import pickle
import json 
from multiprocessing import Queue
from config import cfg

import lib.utils as utils
import lib.accuracy as ut
import lib.generators as gn
import models
from lib.custom_losses import Metric_Loss ,LBA_Loss
from lib.data_process_encoder import LBADataProcess
from models.Encoders import CNNRNNTextEncoder, ShapeEncoder
from multiprocessing import Process, Event 
import time

from torch.utils.tensorboard import SummaryWriter

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from lib.DataLoader import ShapeNetDataset, ShapeNetDataset_Validation, collate_wrapper_val, collate_wrapper_train




	


def retrieval(retrieval_queue,retrieval_proc,text_encoder,embeddings_trained,opts,num,embeddings,cat_mod_id):
	#text_encoder.load_state_dict(torch.load('models/txt_enc.pth'))
	#shape_encoder.load_state_dict(torch.load('models/shape_enc.pth'))
	text_encoder.eval() 
	it = retrieval_proc[0].iters_per_epoch
	iteration = 0
	#################################
	##only 200 elements in validation
	#################################
	
	print("start retrieval")
	#while(iteration < it):
	minibatch = retrieval_queue.get()
	random_index = num
	
	raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch'][random_index]).long().unsqueeze(0).cuda()
	#print(raw_embedding_batch.size())
	txt_embedd_output = text_encoder(raw_embedding_batch)
	#shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
	cat = minibatch['category_list']
	m_id = minibatch['model_list']

	embeddings = np.append(embeddings,txt_embedd_output.detach().cpu().numpy(),axis=0)
	n_neighbors = 4
		#standart euclidean distance
	#nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(embeddings)
	
	nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(embeddings)
	#nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='brute',metric='mahalanobis',metric_params={'V': np.cov(embeddings)}).fit(embeddings)
	distances, indices = nbrs.kneighbors(embeddings)
	queried_indices = indices[-1,:]#the one we are looking for. contain first, scn, third neirest neighbor of the input sentence
	#if(queried_indices[0] == indices.shape[0]-1):
	#	queried_indices = [queried_indices[i] for i in range(len(queried_indices))]

	print(queried_indices)
	first_NN = cat_mod_id[queried_indices[2]]
	print(first_NN)
	print(utils.convert_idx_to_words(minibatch['raw_embedding_batch'][random_index]))
	png_file = opts.png_dir % (first_NN[1], first_NN[1])
	img=mpimg.imread(png_file)
	imgplot = plt.imshow(img)
	plt.show()


def val(text_encoder, shape_encoder,opts):
    text_encoder.eval() 
    shape_encoder.eval() 
    
    #----------------------------------------------------------------------------------------------------
    ## ONLY FOR TEXT AND SHAPE EMBEDDING together
   
    embedding_tuples=[]
    
    generator_val = gn.SS_generator(opts.val_inputs_dict, opts)    	
    for step, minibatch in enumerate(generator_val):

        #raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
        #print("SIZE : ",shape_batch.size())
        #text_encoder_outputs = text_encoder(raw_embedding_batch)
        shape_encoder_outputs = shape_encoder(shape_batch)
        for i in range(opts.batch_size):  
        	embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu()))
    
    
    generator_val = gn.TS_generator(opts.val_inputs_dict, opts)    	
    for step, minibatch in enumerate(generator_val):

        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
        #shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
        #print("SIZE : ",shape_batch.size())
        text_encoder_outputs = text_encoder(raw_embedding_batch)
        #shape_encoder_outputs = shape_encoder(shape_batch)
        for i in range(opts.batch_size):  
        	embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))
        	

    """
    batch_size = 100
    params = {'batch_size':batch_size,
          'shuffle': True,
          'num_workers': 3,
          'collate_fn' : collate_wrapper_val}

    num_of_cap_per_model = 6
    dat_loader = ShapeNetDataset_Validation(opts.val_inputs_dict,num_of_cap_per_model,opts)
    validation_generator = torch.utils.data.DataLoader(dat_loader, **params)
    print("start validation ")
    for captions,shapes,model_ids,categories,labels in validation_generator:
    	print(captions.size())
    	#print(model_ids[0])
    	#print(model_ids[model_ids==model_ids[0]])
    	#input()
    	raw_embedding_batch = captions.long().cuda()
    	shape_batch = shapes.permute(0,4,1,2,3).cuda()
    	text_encoder_outputs = text_encoder(raw_embedding_batch)
    	shape_encoder_outputs = shape_encoder(shape_batch)
    	for i in range(raw_embedding_batch.size()[0]):
    		embedding_tuples.append((categories[i],model_ids[i],text_encoder_outputs[i].data.cpu()))
    		if(i < shape_encoder_outputs.size()[0]):
    			#print(shape_encoder_outputs.size())
    			embedding_tuples.append((categories[i],model_ids[int(i*num_of_cap_per_model)],shape_encoder_outputs[i].data.cpu()))



	
    #caption, category, model_id, embedding
	"""
    outputs_dict = {'caption_embedding_tuples': embedding_tuples, 
                    'dataset_size': len(embedding_tuples)} 

    print("length T-S Embeddings",len(embedding_tuples))
    n_neighbors = 10#6
    print("comp metric")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 

    #precision = np.mean(metrics[0])
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))
    #----------------------------------------------------------------------------------------------------

    return metrics#precision
   
   


def main():
	
	load = True
	parser = argparse.ArgumentParser(description='main text2voxel train/test file')
	parser.add_argument('--dataset',help='dataset', default='shapenet', type=str)
	parser.add_argument('--tensorboard', type=str, default='results')
	opts = parser.parse_args()
	opts.dataset = 'shapenet'
	opts.batch_size = 256
	opts.data_dir = cfg.DIR.RGB_VOXEL_PATH
	opts.png_dir = cfg.DIR.RGB_PNG_PATH
	opts.num_workers = cfg.CONST.NUM_WORKERS
	opts.LBA_n_captions_per_model = 2 #as stated in the paper
	opts.LBA_model_type = cfg.LBA.MODEL_TYPE
	opts.LBA_cosin_dist = cfg.LBA.COSINE_DIST
	opts.rho = cfg.LBA.METRIC_MULTIPLIER
	opts.learning_rate = cfg.TRAIN.LEARNING_RATE

	
	
	
	
	writer = SummaryWriter(os.path.join(opts.tensorboard,'new_new'))
	#we basiaclly neglectthe problematic ones later in the dataloader
	opts.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
	opts_val = copy.deepcopy(opts)
	opts_val.batch_size = 256
	print('----------------- CONFIG -------------------')
	
	#DATA LOADINGDOT_AGAIN_EMBEDDINGS
	######################################################
	inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH) ###DIR.PRIMITIVES_TRAIN_DATA_PATH
	val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH )#DIR.PRIMITIVES_VAL_DATA_PATH
	
	#ret_inputs_dict = utils.open_pickle(cfg.DIR.TEST_DATA_PATH)#processed_captions_test.p
	opts_val.val_inputs_dict = val_inputs_dict#inputs_dict#

	"""
	print("keys :",val_inputs_dict['caption_tuples'][0])
	seen_categories = []
	different_mod_ids = []

	for k in val_inputs_dict['caption_tuples']:
		if(k[2] in different_mod_ids):
			continue
		else:
			different_mod_ids.append(k[2])
		#if(k[1] in seen_categories):
		#	continue
		#else:
		#	seen_categories.append(k[1])

	#print(len(seen_categories))
	print(len(different_mod_ids))
	#return
	"""

	###############################################
	#Test purposes
	#inputs_dict = val_inputs_dict
	#ret_inputs_dict = val_inputs_dict
	#val_inputs_dict = ret_inputs_dict
	###############################################


	#embeddings_trained = utils.open_pickle('test_embeddings.p')
	#data_process_for_class = LBADataProcess
	#val_data_process_for_class = LBADataProcess
	#ret_data_process = LBADataProcess #retrieval process
	#global train_queue, train_processes
	#global val_queue, val_processes
	#global ret_queue, ret_processes
	#queue_capacity = cfg.CONST.QUEUE_CAPACITY
	#queue_capacity = 100

	#train_queue = Queue(queue_capacity)
	#train_processes = utils.make_data_processes(data_process_for_class, train_queue, inputs_dict, opts, repeat=True)
	#val_queue = Queue(queue_capacity)
	#val_processes = utils.make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, opts_val, repeat=True)
	#ret_queue = Queue(100)
	#ret_processes = utils.make_data_processes(ret_data_process,ret_queue,ret_inputs_dict,opts,repeat=False)
	#--------------------------------------------------------------
	

	text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
	shape_encoder = ShapeEncoder().cuda()
	#if(load):
		#text_encoder.load_state_dict(torch.load('models_metric_loss/txt_enc.pth'))
		#shape_encoder.load_state_dict(torch.load('models_metric_loss/shape_enc.pth'))

		#text_encoder.load_state_dict(torch.load('models_sts_tst/txt_enc_loss.pth'))
		#shape_encoder.load_state_dict(torch.load('models_sts_tst/shape_enc_loss.pth'))
	#	text_encoder.load_state_dict(torch.load('models/txt_enc_loss.pth'))
	#	shape_encoder.load_state_dict(torch.load('models/shape_enc_loss.pth'))

	#val(text_encoder,shape_encoder,opts_val)
	#return
	#utils.kill_processes(val_queue, val_processes)
	#return
	#embeddings,cat_mod_id = utils.create_embedding_tuples(embeddings_trained)
	#for i in range(0,200,3):
	#	retrieval(val_queue,val_processes,text_encoder,embeddings_trained,opts,i,embeddings,cat_mod_id)
	
	loss_Metric = Metric_Loss(opts, LBA_inverted_loss=cfg.LBA.INVERTED_LOSS, LBA_normalized=cfg.LBA.NORMALIZE, LBA_max_norm=cfg.LBA.MAX_NORM)

	loss_TST = LBA_Loss(lmbda=0.25, LBA_model_type=cfg.LBA.MODEL_TYPE,batch_size=opts.batch_size) 
	optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#, weight_decay=cfg.TRAIN.DECAY_RATE)
	optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#,weight_decay=cfg.TRAIN.DECAY_RATE)
	#min_batch = train_processes[0].iters_per_epoch
	
	shape_encoder.train()
	text_encoder.train()
	#text_encoder.eval()
	#shape_encoder.eval()
	
	best_loss = np.inf
	params_2 = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 5,
          'collate_fn' : collate_wrapper_train}
	best_acc = 0.0
	dat_loader = ShapeNetDataset(inputs_dict,2,opts)
	training_generator = torch.utils.data.DataLoader(dat_loader, **params_2)
	#mat = []
	for epoch in range(1000):
		print("NEW EPOCH : ",epoch)
		epoch_loss = []
		epoch_loss_STS = []
		epoch_loss_TST = []
		for captions,shapes,model_ids,categories,labels in training_generator:

			#minibatch = train_queue.get()
			#print(minibatch['caption_label_batch'])
			#print("minibatch_shape",minibatch['caption_label_batch'].shape)
			#return
			raw_embedding_batch = captions.long().cuda()#torch.Size([batch_size*2, 96])
			#print(raw_embedding_batch.size())
			shape_batch = shapes.permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
			text_encoder_outputs = text_encoder(raw_embedding_batch)
			shape_encoder_outputs = shape_encoder(shape_batch)
			caption_labels_batch = labels.long().cuda()

			#mat.append((raw_embedding_batch.data.cpu(),minibatch['category_list'],minibatch['model_list'],shape_encoder_outputs.data.cpu()))
			metric_loss = loss_Metric(text_encoder_outputs, shape_encoder_outputs)
			TST_loss, STS_loss = loss_TST(text_encoder_outputs, shape_encoder_outputs,caption_labels_batch) 
			

			complete_loss = (TST_loss+STS_loss) + opts.rho * metric_loss
			
			epoch_loss_STS.append(STS_loss.item())
			epoch_loss_TST.append(TST_loss.item())
			
			epoch_loss.append(complete_loss.item())
			optimizer_text_encoder.zero_grad()
			optimizer_shape_encoder.zero_grad()
			#metric_loss.backward()
			complete_loss.backward()
			optimizer_text_encoder.step()
			optimizer_shape_encoder.step()

		epoch_loss = np.mean(epoch_loss)
		#epoch_loss = 0
		print("LOSS",epoch_loss)
		writer.add_scalar('train loss',epoch_loss,epoch)
		acc = 0
		#if(epoch % 5 == 0):
			#pass
		acc = val(text_encoder,shape_encoder,opts_val)
		writer.add_scalar('validation acc',acc,epoch)
		text_encoder.train()
		shape_encoder.train()
		#print("TST {} and STS {} LOSS MEAN ".format(np.mean(epoch_loss_TST),np.mean(epoch_loss_STS)))

		if(best_acc < acc):
			#best_loss = epoch_loss
			best_acc = acc
			torch.save(text_encoder.state_dict(), 'models/txt_enc_acc.pth')
			torch.save(shape_encoder.state_dict(), 'models/shape_enc_acc.pth')
			print("SAVED MODELS! ACC")
		if(best_loss > epoch_loss):
			best_loss = epoch_loss
			#best_acc = acc
			torch.save(text_encoder.state_dict(), 'models/txt_enc_loss.pth')
			torch.save(shape_encoder.state_dict(), 'models/shape_enc_loss.pth')
			print("SAVED MODELS! LOSS ")

	#utils.create_pickle_embedding(mat)
           
        

    
if __name__ == '__main__':

    main()

"""
def accuracy_test_purpose():
    ################################################################################################################
    model_list = [x for x in range(5)]
    shape_outputs_list = []
    shape_minibatch_list = []


    print(model_list)
    raw_embedding_batch = torch.zeros((10,96)) #10 -batch

    raw_embedding_batch[0] += 1
    raw_embedding_batch[1] += 1.1

    raw_embedding_batch[2] += 20
    raw_embedding_batch[3] += 20.1

    raw_embedding_batch[4] += 30
    raw_embedding_batch[5] += 30.1

    raw_embedding_batch[6] += 40
    raw_embedding_batch[7] += 40.1

    raw_embedding_batch[8] += 50
    raw_embedding_batch[9] += 50.1

    #text_encoder_outputs = torch.zeros((10,2))

    text_encoder_outputs = torch.randn(10,2)
    text_encoder_outputs[0] = 1
    text_encoder_outputs[1] = 1.1

    text_encoder_outputs[2] = 20
    text_encoder_outputs[3] = 20.1

    text_encoder_outputs[4] = 30
    text_encoder_outputs[5] = 30.1

    text_encoder_outputs[6] = 40
    text_encoder_outputs[7] = 40.1

    text_encoder_outputs[8] = 50
    text_encoder_outputs[9] = 50.1


    
    shape_encoder_outputs = torch.randn(5,2)#torch.zeros((5,2))

    shape_encoder_outputs[0] = 1

    shape_encoder_outputs[1] = 20
    shape_encoder_outputs[2] = 30
    shape_encoder_outputs[3] = 40
    shape_encoder_outputs[4] = 50


    minibatch_save = {
            "raw_embedding_batch": raw_embedding_batch,
            'model_list': model_list
        }

    outputs_dict = {
            'text_encoder': text_encoder_outputs, 
            'shape_encoder': shape_encoder_outputs,
            
        }
    shape_outputs_list.append(outputs_dict)
    shape_minibatch_list.append(minibatch_save) 
    
    ####################################################################################################################

    caption_tuples = ut.consolidate_caption_tuples(shape_minibatch_list, shape_outputs_list, opts=None,embedding_type='shape')
    
    outputs_dict = {'caption_embedding_tuples': caption_tuples, 
                    'dataset_size': len(caption_tuples)} 
    metrics = ut.compute_metrics(outputs_dict,n_neighbors = 2) 
    
    #precision = metrics[0] #'precision' 'recall'
    #print(metrics[1])
    print("Mean precision for 5 neighbors : ", metrics)
    #print("Mean precision for 10 neighbors : ", np.mean(precision[0:10]))
    #print("Mean precision for 20 neighbors : ", np.mean(precision[:]))
    input()


    #####################################
    ##########################################################################################
    # COMPUTE TT EMBEDDING SIMILARITY
    shape_minibatch_list = []
    shape_outputs_list = []

    generator_val = ut.TT_generator(opts.val_inputs_dict, opts)
    embedding_tuples=[]
    for step, minibatch in enumerate(generator_val):
        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
        text_encoder_outputs = text_encoder(raw_embedding_batch)
        
        for i in range(opts.batch_size):  
        	#print((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))
        	
        	embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))

    outputs_dict = {'caption_embedding_tuples': embedding_tuples, 
                    'dataset_size': len(embedding_tuples)} 

    print("length T-T Embeddings",len(embedding_tuples))
    n_neighbors = 4
    print("comp metric")
    metrics = ut.compute_metrics(outputs_dict,n_neighbors = n_neighbors) 
    print("precision for Text-to-Text Embeddings with a minimum of {0} matches is : {1}".format(n_neighbors,metrics))
    ################################################################################################
    
"""

"""
    for k in range(min_batch):
    	minibatch = val_queue.get()
    	#raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
    	#print("caption label btach ",minibatch['caption_label_batch'])
    	#print("category list ",minibatch['category_list'])
    	#print(len(minibatch['caption_label_batch']))
    	#print(len(minibatch['category_list']))
    	#input()
    	shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
    	#text_encoder_outputs = text_encoder(raw_embedding_batch)
    	shape_encoder_outputs = shape_encoder(shape_batch)
    	for i in range(opts.batch_size): 
    		#embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu().numpy(),minibatch['raw_embedding_batch'][i]))
        	embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu().numpy()))
    """

"""
    generator_val = gn.SS_generator(opts.val_inputs_dict, opts)    	
    for step, minibatch in enumerate(generator_val):

        #raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
        #print("SIZE : ",shape_batch.size())
        #text_encoder_outputs = text_encoder(raw_embedding_batch)
        shape_encoder_outputs = shape_encoder(shape_batch)
        for i in range(opts.batch_size):  
        	#embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu().numpy(),minibatch['raw_embedding_batch'][i]))
        	embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu()))
    
    
    generator_val = gn.TS_generator(opts.val_inputs_dict, opts)    	
    for step, minibatch in enumerate(generator_val):

        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()
        #shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
        #print("SIZE : ",shape_batch.size())
        text_encoder_outputs = text_enscoder(raw_embedding_batch)
        #shape_encoder_outputs = shape_encoder(shape_batch)
        for i in range(opts.batch_size):  
        	embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))
        	#embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu().numpy()))
    

"""