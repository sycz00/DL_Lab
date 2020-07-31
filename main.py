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
#from lib.data_process_encoder import LBADataProcess
from models.Encoders import CNNRNNTextEncoder, ShapeEncoder
from multiprocessing import Process, Event 
import time

from torch.utils.tensorboard import SummaryWriter

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from lib.DataLoader import ShapeNetDataset,  collate_wrapper_val, collate_wrapper_train




#mat.append((raw_embedding_batch.data.cpu(),minibatch['category_list'],minibatch['model_list'],shape_encoder_outputs.data.cpu()))	
def create_embeddings_pickle(text_encoder, shape_encoder,opts,shape_raw,shape_mod,text_raw,text_mod):
	text_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/txt_enc_acc.pth'))
	shape_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/shape_enc_acc.pth'))

	text_encoder.eval()
	shape_encoder.eval()

	embedding_tuples=[]
	generator_val = gn.generator_minibatch(shape_raw,shape_mod,opts)
	for step, minibatch in enumerate(generator_val):
		shape_batch = torch.from_numpy(minibatch['input']).permute(0,4,1,2,3).cuda()
		shape_encoder_outputs = shape_encoder(shape_batch)
		for i in range(shape_batch.size()[0]):
			embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu()))


	utils.create_pickle_embedding(embedding_tuples,embedd_type ='shape_only')
	
	embedding_tuples_2 = []
	generator_val = gn.generator_minibatch(text_raw,text_mod,opts)
	for step, minibatch in enumerate(generator_val):
		raw_embedding_batch = torch.from_numpy(minibatch['input']).long().cuda()
		text_encoder_outputs = text_encoder(raw_embedding_batch)
		for i in range(raw_embedding_batch.size()[0]):
			embedding_tuples_2.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu(),raw_embedding_batch[i].data.cpu()))
			embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))

	utils.create_pickle_embedding(embedding_tuples_2, embedd_type ='text_only')
	utils.create_pickle_embedding(embedding_tuples,embedd_type ='text_and_shape')



def retrieval(text_encoder,shape_encoder,ret_dict,opts,ret_type='text_to_shape'):
	text_encoder.load_state_dict(torch.load('MODELS/not_normalized_normpenalty/txt_enc_acc.pth'))
	shape_encoder.load_state_dict(torch.load('MODELS/not_normalized_normpenalty/shape_enc_acc.pth'))
	text_encoder.eval() 
	shape_encoder.eval()
	
	
	if(ret_type=='text_to_shape'):
		embeddings_trained = utils.open_pickle('shape_only.p')
		embeddings, model_ids = utils.create_embedding_tuples(embeddings_trained,embedd_type='shape')
	
		queried_captions = []

		print("start retrieval")
		num_of_captions = len(ret_dict['caption_tuples'])
		num_of_retrieval = 10
		iteration = 0
		while(iteration < num_of_retrieval):
		
			rand_ind = np.random.randint(0,num_of_captions)
			caption_tuple = ret_dict['caption_tuples'][rand_ind]
			caption = caption_tuple[0]
			model_id = caption_tuple[2]
			queried_captions.append(caption)
	
			input_caption = torch.from_numpy(caption).unsqueeze(0).long().cuda()
		
	
			txt_embedd_output = text_encoder(input_caption)

			#add the embedding to the trained embeddings
			embeddings = np.append(embeddings,txt_embedd_output.detach().cpu().numpy(),axis=0)
			model_ids.append(model_id)
		
			iteration += 1

	
		n_neighbors = 10
		model_ids = np.array(model_ids)
		embeddings_fused = [(i,j) for i,j in zip(model_ids,embeddings)]
		outputs_dict = {'caption_embedding_tuples': embeddings_fused, 
                    'dataset_size': len(embeddings_fused)} 

		(embeddings_matrix, labels, num_embeddings, label_counter) = ut.construct_embeddings_matrix(outputs_dict)
		indices = ut._compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)
		important_indices = indices[-num_of_retrieval::]
		important_model_id = model_ids[-num_of_retrieval::]

		caption_file = open('Retrieval/text_to_shape/inp_captions.txt', 'w')
		for q in range(num_of_retrieval):
		
			cur_model_id = important_model_id[q]
			all_nn = important_indices[q]
		
			NN = np.argwhere(model_ids[all_nn] == cur_model_id)
			print(" found correct one :",NN)
			if(len(NN) < 1):
				NN = important_indices[q][0]
			else:
				NN = NN[0][0]
				NN = important_indices[q][NN]

			q_caption = queried_captions[q]
			q_model_id = embeddings_fused[NN][0]
			sentence = utils.convert_idx_to_words(q_caption)
			caption_file.write('{}\n'.format(sentence))

			voxel_file = opts.png_dir % (q_model_id,q_model_id)
			img = mpimg.imread(voxel_file)
			imgplot = plt.imshow(img)
			plt.savefig('Retrieval/text_to_shape/{0}.png'.format(q))
			plt.clf()

		caption_file.close()


	elif(ret_type=='shape_to_text'):

		embeddings_trained = utils.open_pickle('text_only.p')
		embeddings, model_ids, raw_caption = utils.create_embedding_tuples(embeddings_trained,embedd_type='text')
	
		queried_shapes = []

		print("start retrieval")
		num_of_captions = len(ret_dict['caption_tuples'])
		num_of_retrieval = 10
		iteration = 0
		while(iteration < num_of_retrieval):
		
			rand_ind = np.random.randint(0,num_of_captions)
			caption_tuple = ret_dict['caption_tuples'][rand_ind]
			#caption = caption_tuple[0]
			model_id = caption_tuple[2]

			cur_shape = utils.load_voxel(None, model_id, opts)
			queried_shapes.append(cur_shape)
			input_shape = torch.from_numpy(cur_shape).unsqueeze(0).permute(0,4,1,2,3).float().cuda()
			#input_caption = torch.from_numpy(caption).unsqueeze(0).long().cuda()
		
	
			#txt_embedd_output = text_encoder(input_caption)
			shape_embedd_output = shape_encoder(input_shape)

			#add the embedding to the trained embeddings
			embeddings = np.append(embeddings,shape_embedd_output.detach().cpu().numpy(),axis=0)
			model_ids.append(model_id)
		
			iteration += 1

	
		n_neighbors = 10
		model_ids = np.array(model_ids)
		embeddings_fused = [(i,j) for i,j in zip(model_ids,embeddings)]
		outputs_dict = {'caption_embedding_tuples': embeddings_fused, 
                    'dataset_size': len(embeddings_fused)} 

		(embeddings_matrix, labels, num_embeddings, label_counter) = ut.construct_embeddings_matrix(outputs_dict)
		indices = ut._compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)
		important_indices = indices[-num_of_retrieval::]
		important_model_id = model_ids[-num_of_retrieval::]

		caption_file = open('Retrieval/shape_to_text/inp_captions.txt', 'w')
		for q in range(num_of_retrieval):
		
			cur_model_id = important_model_id[q]
			all_nn = important_indices[q]
		
			NN = np.argwhere(model_ids[all_nn] == cur_model_id)
			print(" found correct one :",NN)
			if(len(NN) < 1):
				NN = important_indices[q][0]
			else:
				NN = NN[0][0]
				NN = important_indices[q][NN]

			q_shape = queried_shapes[q]
			q_caption = raw_caption[NN].data.numpy()
			q_model_id = embeddings_fused[NN][0]
			sentence = utils.convert_idx_to_words(q_caption)
			caption_file.write('{}\n'.format(sentence))

			voxel_file = opts.png_dir % (q_model_id,q_model_id)
			img = mpimg.imread(voxel_file)
			imgplot = plt.imshow(img)
			plt.savefig('Retrieval/shape_to_text/{0}.png'.format(q))
			plt.clf()

		caption_file.close()
	elif(ret_type=='shape_to_shape'):
		embeddings_trained = utils.open_pickle('shape_only.p')
		embeddings, model_ids = utils.create_embedding_tuples(embeddings_trained,embedd_type='shape')
	
		queried_shapes = []

		print("start retrieval")
		num_of_captions = len(ret_dict['caption_tuples'])
		num_of_retrieval = 10
		iteration = 0
		while(iteration < num_of_retrieval):
		
			rand_ind = np.random.randint(0,num_of_captions)
			caption_tuple = ret_dict['caption_tuples'][rand_ind]
			#caption = caption_tuple[0]
			model_id = caption_tuple[2]
			#queried_captions.append(caption)
	
			cur_shape = utils.load_voxel(None, model_id, opts)
			queried_shapes.append(model_id)
			input_shape = torch.from_numpy(cur_shape).unsqueeze(0).permute(0,4,1,2,3).float().cuda()
			shape_embedd_output = shape_encoder(input_shape)


			#add the embedding to the trained embeddings
			embeddings = np.append(embeddings,shape_embedd_output.detach().cpu().numpy(),axis=0)
			model_ids.append(model_id)
		
			iteration += 1

	
		n_neighbors = 10
		model_ids = np.array(model_ids)
		embeddings_fused = [(i,j) for i,j in zip(model_ids,embeddings)]
		outputs_dict = {'caption_embedding_tuples': embeddings_fused, 
                    'dataset_size': len(embeddings_fused)} 

		(embeddings_matrix, labels, num_embeddings, label_counter) = ut.construct_embeddings_matrix(outputs_dict)
		indices = ut._compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)
		important_indices = indices[-num_of_retrieval::]
		important_model_id = model_ids[-num_of_retrieval::]

		
		for q in range(num_of_retrieval):
		
			cur_model_id = important_model_id[q]
			all_nn = important_indices[q]
		
			NN = np.argwhere(model_ids[all_nn] == cur_model_id)
			print(" found correct one :",NN)
			if(len(NN) < 1):
				NN = important_indices[q][0]
			else:
				NN = NN[0][0]
				NN = important_indices[q][NN]

			
			q_model_id = embeddings_fused[NN][0]
			q_shape = queried_shapes[q] #this is the model id of quieried shape
			

			voxel_file = opts.png_dir % (q_model_id,q_model_id)
			img = mpimg.imread(voxel_file)
			imgplot = plt.imshow(img)
			plt.savefig('Retrieval/shape_to_shape/ret_{0}.png'.format(q))
			plt.clf()

			voxel_file = opts.png_dir % (q_shape,q_shape)
			img = mpimg.imread(voxel_file)
			imgplot = plt.imshow(img)
			plt.savefig('Retrieval/shape_to_shape/inp_{0}.png'.format(q))
			plt.clf()

		










def val(text_encoder, shape_encoder,opts,shape_raw,shape_mod,text_raw,text_mod):
    text_encoder.eval() 
    shape_encoder.eval() 
    
    #----------------------------------------------------------------------------------------------------
    ## ONLY FOR TEXT AND SHAPE EMBEDDING together
   
    embedding_tuples=[]
    generator_val = gn.generator_minibatch(shape_raw,shape_mod,opts)   	
    for step, minibatch in enumerate(generator_val):
        shape_batch = torch.from_numpy(minibatch['input']).permute(0,4,1,2,3).cuda()
        shape_encoder_outputs = shape_encoder(shape_batch)
        for i in range(shape_batch.size()[0]):  
        	embedding_tuples.append((minibatch['model_list'][i],shape_encoder_outputs[i].data.cpu()))
    
    
    generator_val = gn.generator_minibatch(text_raw,text_mod,opts)    	
    for step, minibatch in enumerate(generator_val):
        raw_embedding_batch = torch.from_numpy(minibatch['input']).long().cuda()
        text_encoder_outputs = text_encoder(raw_embedding_batch)
        
        for i in range(raw_embedding_batch.size()[0]):  
        	embedding_tuples.append((minibatch['model_list'][i],text_encoder_outputs[i].data.cpu()))
        	

    
    outputs_dict = {'caption_embedding_tuples': embedding_tuples, 
                    'dataset_size': len(embedding_tuples)} 

    print("length Embeddings",len(embedding_tuples))


    n_neighbors = 20#6
    print("comp metric 10")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))

    n_neighbors = 10#6
    print("comp metric 10")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))

    n_neighbors = 5#6
    print("comp metric 5")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 

    
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))
    #----------------------------------------------------------------------------------------------------

    return metrics
   
   


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
	opts.rho = .5#cfg.LBA.METRIC_MULTIPLIER
	opts.learning_rate = cfg.TRAIN.LEARNING_RATE

	
	
	
	
	writer = SummaryWriter(os.path.join(opts.tensorboard,'Encoder_no_norm_metric_lba_full'))
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

	
	

	text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
	shape_encoder = ShapeEncoder().cuda()

	shape_gen_raw,shape_mod_list = gn.SS_generator(opts_val.val_inputs_dict, opts)
	text_gen_raw,text_mod_list = gn.TS_generator(opts_val.val_inputs_dict, opts)

	#create_embeddings_pickle(text_encoder,shape_encoder,opts_val,shape_gen_raw,shape_mod_list,text_gen_raw,text_mod_list)
	
	if(load):
		#text_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/txt_enc_acc.pth'))
		#shape_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/shape_enc_acc.pth'))

		text_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/txt_enc_loss.pth'))
		shape_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/shape_enc_loss.pth'))

		#text_encoder.load_state_dict(torch.load('MODELS/not_normalized_BOTH_tst_sts/txt_enc_acc.pth'))
		#shape_encoder.load_state_dict(torch.load('MODELS/not_normalized_BOTH_tst_sts/shape_enc_acc.pth'))


	#retrieval(text_encoder,shape_encoder,val_inputs_dict,opts_val,ret_type='text_to_shape')
	#return
	val(text_encoder,shape_encoder,opts_val,shape_gen_raw,shape_mod_list,text_gen_raw,text_mod_list)
	
	return

	
	
	loss_Metric = Metric_Loss(opts, LBA_inverted_loss=cfg.LBA.INVERTED_LOSS, LBA_normalized=cfg.LBA.NORMALIZE, LBA_max_norm=cfg.LBA.MAX_NORM)

	loss_TST = LBA_Loss(lmbda=0.25, LBA_model_type=cfg.LBA.MODEL_TYPE,batch_size=opts.batch_size) 
	optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#, weight_decay=cfg.TRAIN.DECAY_RATE)
	optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#,weight_decay=cfg.TRAIN.DECAY_RATE)
	#min_batch = train_processes[0].iters_per_epoch
	
	shape_encoder.train()
	text_encoder.train()
	#text_encoder.eval()
	#shape_encoder.eval()
	
	best_loss = np.inf8
	params_2 = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 4,
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

			
			raw_embedding_batch = captions.long().cuda()#torch.Size([batch_size*2, 96])
			
			shape_batch = shapes.permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
			text_encoder_outputs = text_encoder(raw_embedding_batch)
			shape_encoder_outputs = shape_encoder(shape_batch)
			caption_labels_batch = labels.long().cuda()

			
			metric_loss = loss_Metric(text_encoder_outputs, shape_encoder_outputs)
			TST_loss,STS_loss = loss_TST(text_encoder_outputs, shape_encoder_outputs,caption_labels_batch) 
			

			complete_loss =  (TST_loss+STS_loss) + opts.rho * metric_loss
			
			#epoch_loss_STS.append(STS_loss.item())
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
		acc = val(text_encoder,shape_encoder,opts_val,shape_gen_raw,shape_mod_list,text_gen_raw,text_mod_list)
		writer.add_scalar('validation acc',acc,epoch)
		text_encoder.train()
		shape_encoder.train()
		print("TST {} and STS LOSS MEAN ".format(np.mean(epoch_loss_TST)))#,np.mean(epoch_loss_STS)

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