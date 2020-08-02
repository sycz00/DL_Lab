import argparse
import numpy as np
import os

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import pickle
import json 
from config import cfg

import lib.utils as utils
import lib.accuracy as ut
import lib.generators as gn
from lib.DataLoader import ShapeNetDataset,  collate_wrapper_val, collate_wrapper_train
from lib.custom_losses import Metric_Loss ,LBA_Loss
from models.Encoders import CNNRNNTextEncoder, ShapeEncoder







	
def create_embeddings_pickle(text_encoder, shape_encoder,opts,shape_raw,shape_mod,text_raw,text_mod):
	#text_encoder.load_state_dict(torch.load('MODELS/METRIC_and_TST/txt_enc_acc.pth'))
	#shape_encoder.load_state_dict(torch.load('MODELS/METRIC_and_TST/shape_enc_acc.pth'))

	text_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/txt_enc_acc.pth'))
	shape_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/shape_enc_acc.pth'))

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
	#text_encoder.load_state_dict(torch.load('MODELS/METRIC_and_TST/txt_enc_acc.pth'))
	#shape_encoder.load_state_dict(torch.load('MODELS/METRIC_and_TST/shape_enc_acc.pth'))

	text_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/txt_enc_acc.pth'))
	shape_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/shape_enc_acc.pth'))

	text_encoder.eval() 
	shape_encoder.eval()
	
	num_of_retrieval = 50
	n_neighbors = 20

	if(ret_type=='text_to_shape'):
		embeddings_trained = utils.open_pickle('shape_only.p')
		embeddings, model_ids = utils.create_embedding_tuples(embeddings_trained,embedd_type='shape')
		length_trained = len(model_ids)
		queried_captions = []

		print("start retrieval")
		num_of_captions = len(ret_dict['caption_tuples'])
		
		iteration = 0
		while(iteration < num_of_retrieval):
		
			#rand_ind = np.random.randint(0,num_of_captions)
			rand_ind = iteration
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

	
		#n_neighbors = 10
		model_ids = np.array(model_ids)
		embeddings_fused = [(i,j) for i,j in zip(model_ids,embeddings)]
		outputs_dict = {'caption_embedding_tuples': embeddings_fused, 
                    'dataset_size': len(embeddings_fused)} 

		(embeddings_matrix, labels, num_embeddings, label_counter) = ut.construct_embeddings_matrix(outputs_dict)
		indices = ut._compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)
		important_indices = indices[-num_of_retrieval::]
		important_model_id = model_ids[-num_of_retrieval::]

		caption_file = open('Retrieval/text_to_shape/inp_captions.txt', 'w')
		counter = 0
		for q in range(num_of_retrieval):
		
			cur_model_id = important_model_id[q]
			all_nn = important_indices[q]
			#kick out all neighbors which are in the queried ones
			all_nn = [n for n in all_nn if n < length_trained]

			NN = np.argwhere(model_ids[all_nn] == cur_model_id)
			print(" found correct one :",NN)
			if(len(NN) < 1):
				NN = important_indices[q][0]
			else:
				counter += 1
				NN = NN[0][0]
				NN = important_indices[q][NN]


			q_caption = queried_captions[q]
			sentence = utils.convert_idx_to_words(q_caption)
			caption_file.write('{}\n'.format(sentence))
			try:
				os.mkdir('Retrieval/text_to_shape/{0}/'.format(q))
			except OSError:
				pass
			for ii,nn in enumerate(all_nn):
				q_model_id = embeddings_fused[nn][0]
				voxel_file = opts.png_dir % (q_model_id,q_model_id)
				img = mpimg.imread(voxel_file)
				imgplot = plt.imshow(img)

				plt.savefig('Retrieval/text_to_shape/{0}/{1}.png'.format(q,ii))
				plt.clf()

			#q_caption = queried_captions[q]
			#q_model_id = embeddings_fused[NN][0]
			#sentence = utils.convert_idx_to_words(q_caption)
			#caption_file.write('{}\n'.format(sentence))

			#voxel_file = opts.png_dir % (q_model_id,q_model_id)
			#img = mpimg.imread(voxel_file)
			#imgplot = plt.imshow(img)
			#plt.savefig('Retrieval/text_to_shape/{0}.png'.format(q))
			#plt.clf()

		caption_file.close()
		print("ACC :",(counter / num_of_retrieval))


	elif(ret_type=='shape_to_text'):

		embeddings_trained = utils.open_pickle('text_only.p')
		embeddings, model_ids, raw_caption = utils.create_embedding_tuples(embeddings_trained,embedd_type='text')
	
		queried_shapes = []

		print("start retrieval")
		num_of_captions = len(ret_dict['caption_tuples'])
		#num_of_retrieval = 10
		iteration = 0
		while(iteration < num_of_retrieval):
		
			#rand_ind = np.random.randint(0,num_of_captions)
			rand_ind = iteration
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

	
		
		model_ids = np.array(model_ids)
		embeddings_fused = [(i,j) for i,j in zip(model_ids,embeddings)]
		outputs_dict = {'caption_embedding_tuples': embeddings_fused, 
                    'dataset_size': len(embeddings_fused)} 

		(embeddings_matrix, labels, num_embeddings, label_counter) = ut.construct_embeddings_matrix(outputs_dict)
		indices = ut._compute_nearest_neighbors_cosine(embeddings_matrix, embeddings_matrix, n_neighbors,True)
		important_indices = indices[-num_of_retrieval::]
		important_model_id = model_ids[-num_of_retrieval::]

		caption_file = open('Retrieval/shape_to_text/inp_captions.txt', 'w')
		counter = 0
		for q in range(num_of_retrieval):
		
			cur_model_id = important_model_id[q]
			all_nn = important_indices[q]

			all_nn = [n for n in all_nn if n < len(raw_caption)]

			
		
			NN = np.argwhere(model_ids[all_nn] == cur_model_id)
			print(" found correct one :",NN)
			if(len(NN) < 1):
				NN = all_nn[0]
			else:
				counter += 1
				NN = NN[0][0]
				NN = all_nn[NN]
				

			#---------------------------------------------------------
			q_shape = queried_shapes[q]
			caption_file = open('Retrieval/shape_to_text/inp_captions{0}.txt'.format(q), 'w')
			for ii,nn in enumerate(all_nn):
				
				q_caption = raw_caption[nn].data.numpy()
				#q_model_id = embeddings_fused[nn][0]
				sentence = utils.convert_idx_to_words(q_caption)
				caption_file.write('{}\n'.format(sentence))
			caption_file.close()

			#---------------------------------------------------------


			#q_shape = queried_shapes[q]
			#q_caption = raw_caption[NN].data.numpy()
			#q_model_id = embeddings_fused[NN][0]
			#sentence = utils.convert_idx_to_words(q_caption)
			#caption_file.write('{}\n'.format(sentence))
			q_model_id = cur_model_id
			voxel_file = opts.png_dir % (q_model_id,q_model_id)
			img = mpimg.imread(voxel_file)
			imgplot = plt.imshow(img)
			plt.savefig('Retrieval/shape_to_text/{0}.png'.format(q))
			plt.clf()

		#caption_file.close()
		print("ACC :",(counter / num_of_retrieval))

	


def val(text_encoder, shape_encoder,opts,shape_raw,shape_mod,text_raw,text_mod):
    text_encoder.eval() 
    shape_encoder.eval() 
    
    #----------------------------------------------------------------------------------------------------
    
   
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


    n_neighbors = 20
    print("comp metric 10")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))

    n_neighbors = 10
    print("comp metric 10")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))

    n_neighbors = 5
    print("comp metric 5")
    data_set = "shapenet"
    metrics = ut.compute_metrics(data_set,outputs_dict,n_neighbors = n_neighbors,nm=1) 

    
    print("precision for Text-to-Shape Embeddings {0} matches is : {1}".format(n_neighbors,metrics))
    #----------------------------------------------------------------------------------------------------

    return metrics
   
   
def train(parameter):

	loss_Metric = Metric_Loss(parameter['opts'])
	loss_TST = LBA_Loss(lmbda=0.25,batch_size=parameter['opts'].batch_size) 

	text_encoder = parameter['text_encoder']
	shape_encoder = parameter['shape_encoder']
	optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#, weight_decay=cfg.TRAIN.DECAY_RATE)
	optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#,weight_decay=cfg.TRAIN.DECAY_RATE)
	
	
	shape_encoder.train()
	text_encoder.train()
	
	
	best_loss = np.inf
	params_2 = {'batch_size': parameter['opts'].batch_size,
          'shuffle': True,
          'num_workers': 4,
          'collate_fn' : collate_wrapper_train}
	best_acc = 0.0
	dat_loader = ShapeNetDataset(parameter['inputs_dict'],2,parameter['opts'])
	training_generator = torch.utils.data.DataLoader(dat_loader, **params_2)
	
	for epoch in range(1000):
		print("NEW EPOCH : ",epoch)
		epoch_loss = []
		epoch_loss_STS = []
		epoch_loss_TST = []
		for captions,shapes,model_ids,categories,labels in training_generator:

			
			raw_embedding_batch = captions.long().cuda()
			
			shape_batch = shapes.permute(0,4,1,2,3).cuda() 
			text_encoder_outputs = text_encoder(raw_embedding_batch)
			shape_encoder_outputs = shape_encoder(shape_batch)
			caption_labels_batch = labels.long().cuda()

			
			metric_loss = loss_Metric(text_encoder_outputs, shape_encoder_outputs)
			TST_loss,STS_loss = loss_TST(text_encoder_outputs, shape_encoder_outputs,caption_labels_batch) 
			

			complete_loss =  (TST_loss+STS_loss) + parameter['opts'].rho * metric_loss
			
			
			epoch_loss_TST.append(TST_loss.item())
			
			epoch_loss.append(complete_loss.item())
			optimizer_text_encoder.zero_grad()
			optimizer_shape_encoder.zero_grad()
			
			complete_loss.backward()
			optimizer_text_encoder.step()
			optimizer_shape_encoder.step()

		epoch_loss = np.mean(epoch_loss)
		
		print("LOSS",epoch_loss)
		parameter['writer'].add_scalar('train loss',epoch_loss,epoch)
		acc = val(text_encoder,shape_encoder,parameter['opts_val'],parameter['shape_gen_raw'],parameter['shape_mod_list'],parameter['text_gen_raw'],parameter['text_mod_list'])
		parameter['writer'].add_scalar('validation acc',acc,epoch)
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


def main():
	
	load = True
	parser = argparse.ArgumentParser(description='main text2voxel train/test file')
	parser.add_argument('--dataset',help='dataset', default='shapenet', type=str)
	parser.add_argument('--tensorboard', type=str, default='results')
	parser.add_argument('--batch_size',type=int,default=100)
	parser.add_argument('--data_dir',type=str,default=cfg.DIR.RGB_VOXEL_PATH)
	parser.add_argument('--png_dir',type=str,default=cfg.DIR.RGB_PNG_PATH)
	parser.add_argument('--num_workers',type=int,default=cfg.CONST.NUM_WORKERS)
	parser.add_argument('--LBA_n_captions_per_model',type=int,default=2)
	parser.add_argument('--rho',type=float,default=0.5)
	parser.add_argument('--learning_rate',type=float,default=cfg.TRAIN.LEARNING_RATE)
	parser.add_argument('--probablematic_nrrd_path',type=str,default=cfg.DIR.PROBLEMATIC_NRRD_PATH)
	parser.add_argument('--train',type=bool,default=True)
	parser.add_argument('--retrieval',type=bool,default=False)
	parser.add_argument('--tensorboard_name',type=str,default='test')


	opts = parser.parse_args()
	
	writer = SummaryWriter(os.path.join(opts.tensorboard,opts.tensorboard_name))
	inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH) 
	val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH )
	#we basiaclly neglectthe problematic ones later in the dataloader
	#opts.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
	opts_val = copy.deepcopy(opts)
	opts_val.batch_size = 256
	opts_val.val_inputs_dict = val_inputs_dict


	text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
	shape_encoder = ShapeEncoder().cuda()

	shape_gen_raw,shape_mod_list = gn.SS_generator(opts_val.val_inputs_dict, opts)
	text_gen_raw,text_mod_list = gn.TS_generator(opts_val.val_inputs_dict, opts)

	parameter = {'shape_encoder':shape_encoder, 'text_encoder':text_encoder,'shape_gen_raw':shape_gen_raw,'shape_mod_list':shape_mod_list,
		'text_gen_raw':text_gen_raw,'text_mod_list':text_mod_list,'writer':writer,'opts':opts,'inputs_dict':inputs_dict,'opts_val':opts_val}


	if(opts.train):
		train(parameter)
	
	

	
	
	

	#create_embeddings_pickle(text_encoder,shape_encoder,opts_val,shape_gen_raw,shape_mod_list,text_gen_raw,text_mod_list)
	
	#if(load):
	#	text_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/txt_enc_acc.pth'))
	#	shape_encoder.load_state_dict(torch.load('MODELS/not_norm_metric_tst/shape_enc_acc.pth'))

		#text_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/txt_enc_acc.pth'))
		#shape_encoder.load_state_dict(torch.load('MODELS/METRIC_ONLY/shape_enc_acc.pth'))

		#text_encoder.load_state_dict(torch.load('MODELS/not_normalized_BOTH_tst_sts/txt_enc_acc.pth'))
		#shape_encoder.load_state_dict(torch.load('MODELS/not_normalized_BOTH_tst_sts/shape_enc_acc.pth'))

		#text_encoder.load_state_dict(torch.load('models/txt_enc_acc.pth'))
		#shape_encoder.load_state_dict(torch.load('models/shape_enc_acc.pth'))


	#retrieval(text_encoder,shape_encoder,val_inputs_dict,opts_val,ret_type='text_to_shape')
	#val(text_encoder,shape_encoder,opts_val,shape_gen_raw,shape_mod_list,text_gen_raw,text_mod_list)

           
        

    
if __name__ == '__main__':

    main()

