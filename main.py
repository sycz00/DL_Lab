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
import models
from lib.custom_losses import LBA_Loss 
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

def accuracy(X,Y):
    bsz = X.size(0)
    err_1 = ((X[:bsz//2]-Y)**2).sum(axis=0)
    err_2 = ((X[bsz//2:bsz]-Y)**2).sum(axis=0)
    return err_1+err_2

def val(val_que,val_proces, text_encoder, shape_encoder,loss):
    iterations = val_process[0].iters_per_epoch
    losses = []
    text_encoder.eval() 
    shape_encoder.eval()
    acc = []
    for i in range(iterations):
        minibatch = val_queue.get()
        
        raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()#torch.Size([batch_size*2, 96])
        
        shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
        caption_labels_batch = torch.from_numpy(minibatch['caption_label_batch']).long().cuda()
        
        shape_category_batch = minibatch['category_list']

        ###################
        

        #metric_loss = loss(text_encoder_outputs, shape_encoder_outputs)

        #losses.append(metric_loss)
        minibatch_save = {
            "raw_embedding_batch": raw_embedding_batch.data.cpu(),
            'caption_labels_batch': caption_labels_batch.data.cpu(),
            'category_list': shape_category_batch,
            'model_list': minibatch['model_list']
        }


        text_encoder_outputs = text_encoder(raw_embedding_batch)
        shape_encoder_outputs = shape_encoder(shape_batch)


    return np.mean(np.array(losses)), 0

        
    
def main():
   
    
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

    #val_queue = Queue(queue_capacity)
    #val_processes = make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, opts, repeat=True)

    
    text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
    shape_encoder = ShapeEncoder().cuda()
    
    #val(val_queue,val_processes,text_encoder,shape_encoder,opts)
    #########################################
    #----------------------------------------
    #########################################    

    loss2 = Metric_Loss(opts, LBA_inverted_loss=cfg.LBA.INVERTED_LOSS, LBA_normalized=cfg.LBA.NORMALIZE, LBA_max_norm=cfg.LBA.MAX_NORM)
    loss1 = LBA_Loss(lmbda=0.25, LBA_model_type=cfg.LBA.MODEL_TYPE,batch_size=opts.batch_size)
    

    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#, weight_decay=cfg.TRAIN.DECAY_RATE) 
    optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)#,weight_decay=cfg.TRAIN.DECAY_RATE) 

    min_batch = train_processes[0].iters_per_epoch
    text_encoder.train() 
    shape_encoder.train()

    for epoch in range(1000):
        print("NEW EPOCH")
        epoch_loss = []
        for i in range(min_batch):
            minibatch = train_queue.get()

            raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()#torch.Size([batch_size*2, 96])
            caption_labels_batch = torch.from_numpy(minibatch['caption_label_batch']).long().cuda()

    
            shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
            
            text_encoder_outputs = text_encoder(raw_embedding_batch)
            shape_encoder_outputs = shape_encoder(shape_batch)
           
            
            lba_loss,_,_ = loss1(text_encoder_outputs, shape_encoder_outputs,caption_labels_batch) 
            metric_loss = loss2(text_encoder_outputs, shape_encoder_outputs)
            
            loss = lba_loss + opts.rho * metric_loss
            
            epoch_loss.append(loss.item())
            
                

            optimizer_text_encoder.zero_grad() 
            optimizer_shape_encoder.zero_grad() 
            metric_loss.backward() 

            optimizer_text_encoder.step()
            optimizer_shape_encoder.step() 
        print("LOSS",np.mean(epoch_loss))

        #if(epoch % 10 == 0):
            #vall_loss,val_acc = val(val_queue,val_processes,text_encoder,shape_encoder,loss)

            #text_encoder.train() 
            #shape_encoder.train()
            
        

    
if __name__ == '__main__':
    main()
