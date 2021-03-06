import torch
import torch.nn as nn

import torch.nn.functional as F 


class CNNRNNTextEncoder(nn.Module):

    def __init__(self, vocab_size,embedding_size=128,normalize=False):

        super(CNNRNNTextEncoder,self).__init__()
        self.normalize = normalize
        self.emb = nn.Embedding(vocab_size, embedding_size)
        self.f1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=3//2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1,padding=3//2), 
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=1,padding=3//2), 
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1,padding=3//2),  
            nn.ReLU(),
            nn.BatchNorm1d(256)

        )
        
        self.f5 = nn.GRU(256,256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),#24576
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def compute_sequence_length(self, caption_batch):
        
        seq_length_variable = torch.gt(caption_batch, 0).sum(dim=1)
        seq_length_variable = seq_length_variable.long()
        return seq_length_variable

    def forward(self, x):
        seq_len = self.compute_sequence_length(x)
        #print("seq length:",seq_len)
        max_seq_len = x.size(1)
        x = self.emb(x).permute(0, 2, 1)
        x = self.f1(x).permute(2, 0, 1)
        
        
        x, _ = self.f5(x)
        
        masks = (seq_len-1).unsqueeze(0).unsqueeze(2).expand(max_seq_len, x.size(1), x.size(2))
        x = x.gather(0, masks)[0]
        #x = x.squeeze(0)
        x = self.classifier(x)
        #norm = x.norm(dim=1, p=2, keepdim=True)
        #x = x.div(norm.expand_as(x))
        
        #return F.normalize(x, p=2, dim=1)
    
        return x


class ShapeEncoder(nn.Module):
    def __init__(self,num_channel=4,num_classes=2,normalize=False):
        super(ShapeEncoder, self).__init__()

        self.normalize = normalize
        self.f1 = nn.Sequential(  
            nn.Conv3d(num_channel, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2),padding=(3//2,3//2,3//2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2),padding=(3//2,3//2,3//2)), 
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2),padding=(3//2,3//2,3//2)), 
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AvgPool3d(3,stride=2)
            )
        
        """
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),#24576
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        """
        self.classifier = nn.Linear(256, 128)



    def forward(self, x):
        x = self.f1(x)
       
        
        
        x = x.view(x.size(0), -1)
        #x = F.softmax(self.classifier(x), dim=1)
        x = self.classifier(x)
        
        #return F.normalize(x, p=2, dim=1)
        #norm = x.norm(dim=1, p=2, keepdim=True)
        #x = x.div(norm.expand_as(x))
        
        return x