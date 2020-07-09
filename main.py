import argparse
import numpy as np
import os


import torch
import torch.optim as optim
# HACK: Get logger to print to stdout
import sys



from multiprocessing import Queue
from config import cfg

import utils as utils
import lib.accuracy as ut
import models
from lib.custom_losses import Metric_Loss 
from lib.data_process_encoder import LBADataProcess
from models.Encoders import CNNRNNTextEncoder, ShapeEncoder
from multiprocessing import Process, Event 
import time
from torch.utils.tensorboard import SummaryWriter


def kill_processes(queue, processes): 
    print('signal processes to shutdown')

    for p in processes: 
        p.shutdown() 

    
    
    while not queue.empty(): # If queue is not empty 
        time.sleep(0.5)
        try: 
            queue.get(False) 
        except:
            print('now queue size is {0}'.format(queue.qsize()))
            pass 

    print('killing processes.') 
    for p in processes:
        p.terminate() 

def make_data_processes(data_process_class, queue, data_paths, opts, repeat): 
    
    processes = [] 
    for i in range(opts.num_workers): 
        process = data_process_class(queue, data_paths, opts, repeat=repeat)
        process.start() 
        processes.append(process) 

    return processes 

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
    
    precision = metrics[0] #'precision' 'recall'
    print(metrics[1])
    print("Mean precision for 5 neighbors : ", np.mean(precision))
    #print("Mean precision for 10 neighbors : ", np.mean(precision[0:10]))
    #print("Mean precision for 20 neighbors : ", np.mean(precision[:]))
    input()
    

def val(val_queue,val_process, text_encoder, shape_encoder,opts):
    text_encoder.eval() 
    shape_encoder.eval() 
    #opts.test_or_val_phase = True 
    val_iters_per_epoch = val_process[0].iters_per_epoch
    print("LENGTH VALIDATION SET : " ,val_process[0].num_data)
    iteration = 0
    shape_minibatch_list = []
    shape_outputs_list = []
    ######################################################################
    ## computing shape embeddings 
    ######################################################################
    print('computing validation accuracy')
    while iteration < val_iters_per_epoch:
        
        
        minibatch = val_queue.get()  
        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()


        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda()
        
 
        minibatch_save = {
            "raw_embedding_batch": raw_embedding_batch.data.cpu(),
            'model_list': minibatch['model_list']
        }

        text_encoder_outputs = text_encoder(raw_embedding_batch)
        shape_encoder_outputs = shape_encoder(shape_batch)

       
        outputs_dict = {
            'text_encoder': text_encoder_outputs.data.cpu(), 
            'shape_encoder': shape_encoder_outputs.data.cpu(), 
        }
       
        shape_outputs_list.append(outputs_dict)
        shape_minibatch_list.append(minibatch_save)  

        
        iteration = iteration + 1
   
    

    caption_tuples = ut.consolidate_caption_tuples(shape_minibatch_list, shape_outputs_list, opts,embedding_type='shape')
    
    outputs_dict = {'caption_embedding_tuples': caption_tuples, 
                    'dataset_size': len(caption_tuples)} 
    metrics = ut.compute_metrics(outputs_dict,n_neighbors = 10) 
    
    precision = metrics[0] #[1]'precision' 'recall'
    print("Mean precision for {0} neighbors is : {1}".format(10,np.mean(precision)))
   
    
    
    
def main():
   
    #accuracy()
    parser = argparse.ArgumentParser(description='main text2voxel train/test file')
    parser.add_argument('--dataset',help='dataset', default='shapenet', type=str)
    opts = parser.parse_args()
    opts.dataset = 'shapenet'
    opts.batch_size = 100
    opts.data_dir = cfg.DIR.RGB_VOXEL_PATH
    opts.num_workers = cfg.CONST.NUM_WORKERS
    opts.LBA_n_captions_per_model = 2 #as stated in the paper
    opts.LBA_model_type = cfg.LBA.MODEL_TYPE
    opts.LBA_cosin_dist = cfg.LBA.COSINE_DIST
    opts.rho = cfg.LBA.METRIC_MULTIPLIER
    opts.learning_rate = cfg.TRAIN.LEARNING_RATE
    #writer = SummaryWriter('/logs/')

    #we basiaclly neglectthe problematic ones later in the dataloader
    opts.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
    print('----------------- CONFIG -------------------')

    
    

    inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH) #processed_captions_train.p
    val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH)#processed_captions_val.p
    test_inputs_dict = utils.open_pickle(cfg.DIR.TEST_DATA_PATH)#processed_captions_test.p
    
    
    data_process_for_class = LBADataProcess
    val_data_process_for_class = LBADataProcess
   
    global train_queue, train_processes 
    global val_queue, val_processes 
    queue_capacity = cfg.CONST.QUEUE_CAPACITY 
    #queue_capacity = 20
    train_queue = Queue(queue_capacity)
    
    
    train_processes = make_data_processes(data_process_for_class, train_queue, inputs_dict, opts, repeat=True) 

    val_queue = Queue(queue_capacity)
    val_processes = make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, opts, repeat=True)


    
    text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
    shape_encoder = ShapeEncoder().cuda()

    
    #val(val_queue,val_processes,text_encoder,shape_encoder,opts)
    #########################################
    #----------------------------------------
    #########################################    

    loss = Metric_Loss(opts, LBA_inverted_loss=cfg.LBA.INVERTED_LOSS, LBA_normalized=cfg.LBA.NORMALIZE, LBA_max_norm=cfg.LBA.MAX_NORM)
    
    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.DECAY_RATE) 
    optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE,weight_decay=cfg.TRAIN.DECAY_RATE) 

    min_batch = train_processes[0].iters_per_epoch
    text_encoder.train() 
    shape_encoder.train()

    for epoch in range(1000):
        print("NEW EPOCH : ",epoch)
        epoch_loss = []
        for i in range(min_batch):
            minibatch = train_queue.get()

            raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()#torch.Size([batch_size*2, 96])
            
            shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
            
            text_encoder_outputs = text_encoder(raw_embedding_batch)
            shape_encoder_outputs = shape_encoder(shape_batch)
           
            
            metric_loss = loss(text_encoder_outputs, shape_encoder_outputs) 
            epoch_loss.append(metric_loss.item())
            
                

            optimizer_text_encoder.zero_grad() 
            optimizer_shape_encoder.zero_grad() 
            metric_loss.backward() 

            optimizer_text_encoder.step()
            optimizer_shape_encoder.step() 
        print("LOSS",np.mean(epoch_loss))

        if(epoch % 2 == 0):
        	val(val_queue,val_processes,text_encoder,shape_encoder,opts)
        	text_encoder.train()
        	shape_encoder.train()
            
        

    
if __name__ == '__main__':
    main()
