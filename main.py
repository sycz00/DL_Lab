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
from lib.custom_losses import Metric_Loss 
from lib.data_process_encoder import LBADataProcess
from models.Encoders import CNNRNNTextEncoder, ShapeEncoder
from multiprocessing import Process, Event 
import time


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
def val():
    return 0
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

    #we basiaclly neglectthe problematic ones later in the dataloader
    opts.probablematic_nrrd_path = cfg.DIR.PROBLEMATIC_NRRD_PATH
    print('----------------- CONFIG -------------------')
    #pprint.pprint(cfg)

    
    #SHAPENET DATASET 
    inputs_dict = utils.open_pickle(cfg.DIR.TRAIN_DATA_PATH)#PRIMITIVES_TRAIN_DATA_PATH)#) #processed_captions_train.p
    val_inputs_dict = utils.open_pickle(cfg.DIR.VAL_DATA_PATH)#PRIMITIVES_VAL_DATA_PATH)#processed_captions_val.p
    test_inputs_dict = utils.open_pickle(cfg.DIR.TEST_DATA_PATH)#PRIMITIVES_TEST_DATA_PATH)#processed_captions_test.p

    data_process_for_class = LBADataProcess
    val_data_process_for_class = LBADataProcess
   
    global train_queue, train_processes 
    global val_queue, val_processes 
    queue_capacity = cfg.CONST.QUEUE_CAPACITY 
    train_queue = Queue(queue_capacity)
    
    
    train_processes = make_data_processes(data_process_for_class, train_queue, inputs_dict, opts, repeat=True) 

    #val_queue = Queue(queue_capacity)
    #val_processes = make_data_processes(val_data_process_for_class, val_queue, val_inputs_dict, opts, repeat=True)

    
    text_encoder = CNNRNNTextEncoder(vocab_size=inputs_dict['vocab_size']).cuda()
    shape_encoder = ShapeEncoder().cuda()


    
    #LBA_loss = LBA_Loss(lmbda=0.25, LBA_model_type=opts.LBA_model_type, batch_size=opts.batch_size)

    
    
    Metric_loss = Metric_Loss(opts, LBA_inverted_loss=cfg.LBA.INVERTED_LOSS, LBA_normalized=cfg.LBA.NORMALIZE, LBA_max_norm=cfg.LBA.MAX_NORM)
    

    
    optimizer_text_encoder = optim.Adam(text_encoder.parameters(), lr=opts.learning_rate, weight_decay=cfg.TRAIN.DECAY_RATE) 
    optimizer_shape_encoder = optim.Adam(shape_encoder.parameters(), lr=opts.learning_rate,weight_decay=cfg.TRAIN.DECAY_RATE) 

    min_batch = train_processes[0].iters_per_epoch
    
    for epoch in range(1000):
        print("NEW EPOCH")
        for i in range(min_batch):
            minibatch = train_queue.get()
            #kill_processes(train_queue, train_processes)
            raw_embedding_batch = torch.from_numpy(minibatch['raw_embedding_batch']).long().cuda()#torch.Size([batch_size*2, 96])
            shape_batch = torch.from_numpy(minibatch['voxel_tensor_batch']).permute(0,4,1,2,3).cuda() #torch.Size([batch_size,4,32,32,32])
            text_encoder_outputs = text_encoder(raw_embedding_batch)
            shape_encoder_outputs = shape_encoder(shape_batch)
            text_encoder.train() 
            shape_encoder.train()
            #lba_loss, _, _ = LBA_loss(text_encoder_outputs, shape_encoder_outputs, caption_labels_batch)
            metric_loss = Metric_loss(text_encoder_outputs, shape_encoder_outputs) 
            loss = metric_loss#lba_loss + opts.rho *
            if(i % 50 == 0):
                print("LOSS :",loss.item())

            optimizer_text_encoder.zero_grad() 
            optimizer_shape_encoder.zero_grad() 
            loss.backward() 

            optimizer_text_encoder.step()
            optimizer_shape_encoder.step() 
            
        

    
if __name__ == '__main__':
    main()
