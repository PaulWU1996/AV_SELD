r"""

audio_extractor: 4 residual cnn blocks
visual_extractor: resnet18

crossmodal conformer: 2 layers

ssl predictor: 2 layers
"""
from typing import Any
from torchvision.models import ResNet18_Weights

import torch 
import torch.nn as nn
import numpy as np


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        if stride!=1 or in_dim!=hidden_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, hidden_dim, kernel_size=(1, 1), stride=(stride, stride)),
                nn.BatchNorm2d(hidden_dim)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ConformerBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, num_heads=4, dropout=0.1):
        super(ConformerBlock, self).__init__()
        dim_feedforward = 2 * hidden_dim
        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, x):
        #feedward1
        residual = x
        x = self.feed_forward1(x)
        x = self.norm1(residual*0.5 + self.dropout(x))
        #multi-head attention
        residual = x
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2) 
        x = self.norm2(residual*0.5 + self.dropout(x))
        #convolution
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(residual*0.5 + self.dropout(x))
        #feedward2
        residual = x
        x = self.feed_forward2(x)
        x = self.norm4(residual*0.5 + self.dropout(x))
        return x


class CMConformerBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, num_heads=4, dropout=0.1):
        super(CMConformerBlock, self).__init__()
        dim_feedforward = 2 * hidden_dim
        self.feed_forward1 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)

    def forward(self, x, y):
        r"""
        Input:
            x: main modality's input (batch, time, hidden_dim)
            y: external modality's input (batch, time, hidden_dim)
        Output:
            x: main modality's output (batch, time, hidden_dim)
        """

        # feedward1
        residual = x
        x = self.feed_forward1(x)
        x = self.norm1(residual*0.5 + self.dropout(x))

        # attention part
        residual = x

        x = x.permute(1, 0, 2)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)

        x_ = residual.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        x_, _ = self.cross_attention(x_, y, y)
        x_ = x_.permute(1, 0, 2)

        x = (x + x_) * 0.5
        x = self.norm2(residual*0.5 + self.dropout(x))

        # convolution
        residual = x
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(residual*0.5 + self.dropout(x))

        # feedward2
        residual = x
        x = self.feed_forward2(x)
        x = self.norm4(residual*0.5 + self.dropout(x))
        return x

class CMConformer(nn.Module):
    def __init__(self, hidden_dim, num_confblks=1, kernel_size=3, num_heads=4, dropout=0.1):
        super(CMConformer, self).__init__()
        self.num_confblks = num_confblks
        moduleDict = {}
        for i in range(self.num_confblks):
            moduleDict['cm_a_'+str(i)] = CMConformerBlock(hidden_dim, kernel_size, num_heads, dropout)
            moduleDict['cm_v_'+str(i)] = CMConformerBlock(hidden_dim, kernel_size, num_heads, dropout)
        self.moduleDict = nn.ModuleDict(moduleDict)

    def forward(self, x, y):
        r"""
        Input:
            x: main modality's input (batch, time, hidden_dim)
            y: external modality's input (batch, time, hidden_dim)
        Output:
            x: main modality's output (batch, time, hidden_dim)
        """

        for i in range(self.num_confblks):
            x_, y_ = x, y

            x = self.moduleDict['cm_a_'+str(i)](x_, y_)
            y = self.moduleDict['cm_v_'+str(i)](y_, x_)

        out = torch.concat((x, y), dim=2)
        
        return out

r"""
    只准备一个crossmodal conformer block
"""
class SingleCMConformer(nn.Module):
    def __init__(self, hidden_dim, num_confblks=1, kernel_size=3, num_heads=4, dropout=0.1):
        super(CMConformer, self).__init__()
        self.num_confblks = num_confblks
        moduleDict = {}
        for i in range(self.num_confblks):
            moduleDict['cm_'+str(i)] = CMConformerBlock(hidden_dim, kernel_size, num_heads, dropout)
        self.moduleDict = nn.ModuleDict(moduleDict)

    def forward(self, x, y):
        r"""
        Input:
            x: main modality's input (batch, time, hidden_dim)
            y: external modality's input (batch, time, hidden_dim)
        Output:
            x: main modality's output (batch, time, hidden_dim)
        """

        for i in range(self.num_confblks):
            x_, y_ = x, y

            x = self.moduleDict['cm_'+str(i)](x_, y_)
            y = y_

        out = torch.concat((x, y), dim=2)
        
        return out

class AV_SELD(nn.Module):
    def __init__(self,
                    res_in = [16, 64, 128, 256], res_out = [64, 128, 256, 512], 
                    input_channels=16, n_bins=128, num_classes=14, num_resblks=4, num_confblks=2, hidden_dim=512, kernel_size=3, num_heads=4, dropout=0.1,
                    output_classes=14, class_overlaps=3):
        super(AV_SELD, self).__init__()
        self.n_bins = n_bins
        self.num_classes = num_classes

        self.bn1 = nn.BatchNorm2d(self.n_bins)

        # audio residual blocks
        self.audio_res = self._make_layer(res_in, res_out, num_resblks)
        # self.audio_res = nn.Sequential(
        #     nn.Conv2d(input_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)),
        #     self._make_layer(hidden_dim, num_resblks),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2)),
        # )

        # visual resnet18
        resnet18 = torch.hub.load('pytorch/vision:v0.14.1', 'resnet18', weights=ResNet18_Weights.DEFAULT) # pretrained=False will be changed to True
        modules = list(resnet18.children())[:-2]
        extractor = nn.Sequential(*modules)
        for p in extractor.parameters():
            p.requires_grad = True # False
        self.visual_res = nn.Sequential(
            extractor,
            nn.Flatten(2),
            nn.Linear(49, 80),
            nn.ReLU(),
        )

        # crossmodal conformer
        self.cm_conformer = CMConformer(hidden_dim=hidden_dim, num_confblks=num_confblks, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout)

        # dense layers
        self.dense_layer = nn.Linear(80,10)

        # SSL predictor
        sed_output_size = output_classes * class_overlaps    #here 3 is the max number of simultaneus sounds from the same class
        doa_output_size = sed_output_size * 3   #here 3 is the number of spatial dimensions xyz
        self.sed = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout*3),
                    nn.Linear(1024, sed_output_size),
                    nn.Sigmoid())

        self.doa = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout*3),
                    nn.Linear(1024, doa_output_size),
                    nn.Tanh())

    # def _make_layer(self, in_dim, hidden_dim, num_resblks):
    #     layers = []
    #     for i in range(num_resblks): 
    #         layers.append(ResNetBlock(hidden_dim))
    #     return nn.Sequential(*layers)
    def _make_layer(self, in_dim, hidden_dim, num_resblks):
        layers = []
        for i in range(num_resblks): 
            layers.append(ResNetBlock(in_dim=in_dim[i], hidden_dim=hidden_dim[i],))
        return nn.Sequential(*layers)



    def forward(self, audio, img):

        r"""
        time !!!!! This is not being confirmed yet
        Input:
            audio: torch size ([batch, 16, 128, 2400])
            img: torch size ([batch, 3, 224, 224])

        Output:
            res: torch size ([batch, 14, 128, 300])
        """

        # audio feature extraction
        x = audio.permute(0,2,1,3)
        x = self.bn1(x)
        x = x.permute(0,2,3,1)
        x = self.audio_res(x)
        x = x.mean(3)
        x = x.permute(0,2,1) # torch size ([batch, time, 512]) for conformer input 

        # visual feature extraction
        y = self.visual_res(img)
        y = y.permute(0,2,1) # torch size ([batch, time, 512]) for conformer input

        # crossmodal conformer
        z = self.cm_conformer(x, y) # torch size ([batch, time, 1024])
        
        # dense layers
        z = z.permute(0,2,1)
        z = self.dense_layer(z) # torch size ([batch, 1024, 300])
        z = z.permute(0,2,1)

        # output layer
        sed = self.sed(z) # torch size ([batch, 14*3])
        doa = self.doa(z) # torch size ([batch, 14*3*3])

        return sed, doa


class Test_demo(nn.Module):
    def __init__(self,
                    res_in = [16, 64, 128, 256], res_out = [64, 128, 256, 512], 
                    input_channels=16, n_bins=128, num_classes=14, num_resblks=4, num_confblks=2, hidden_dim=512, kernel_size=3, num_heads=4, dropout=0.1,
                    output_classes=14, class_overlaps=3):
        super(Test_demo, self).__init__()

        self.n_bins = n_bins
        self.num_classes = num_classes

        self.bn1 = nn.BatchNorm2d(self.n_bins)

        # audio residual blocks
        self.audio_res = self._make_layer(res_in, res_out, num_resblks)

        # visual resnet18
        resnet18 = torch.hub.load('pytorch/vision:v0.14.1', 'resnet18', weights=ResNet18_Weights.DEFAULT) # pretrained=False will be changed to True
        modules = list(resnet18.children())[:-2]
        extractor = nn.Sequential(*modules)
        for p in extractor.parameters():
            p.requires_grad = True # False
        self.visual_res = nn.Sequential(
            extractor,
            nn.Flatten(2),
            nn.Linear(49, 2400),
            nn.ReLU(),
        )

        # dense layers
        self.dense_layer = nn.Linear(2400,300)

        # test output branches
         # SSL predictor
        sed_output_size = output_classes * class_overlaps    #here 3 is the max number of simultaneus sounds from the same class
        doa_output_size = sed_output_size * 3   #here 3 is the number of spatial dimensions xyz
        self.sed = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout*3),
                    nn.Linear(1024, sed_output_size),
                    nn.Sigmoid())

        self.doa = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Dropout(dropout*3),
                    nn.Linear(1024, doa_output_size),
                    nn.Tanh())

    def forward(self, audio, img):
        # audio feature extraction
        x = audio.permute(0,2,1,3)
        x = self.bn1(x)
        x = x.permute(0,2,3,1)
        x = self.audio_res(x)
        x = x.mean(3)
        x = x.permute(0,2,1) # torch size ([batch, time, 512]) for conformer input 

        # visual feature extraction
        y = self.visual_res(img)
        y = y.permute(0,2,1) # torch size ([batch, time, 512]) for conformer input

        # concat x & y
        z = torch.cat((x, y), dim=2) # torch size ([batch, time, 1024])

        # dense layers
        z = z.permute(0,2,1)
        z = self.dense_layer(z) # torch size ([batch, 1024, 300])
        z = z.permute(0,2,1)

        # output layer
        sed = self.sed(z) # torch size ([batch, 14*3])
        doa = self.doa(z) # torch size ([batch, 14*3*3])

        return sed, doa

    def _make_layer(self, in_dim, hidden_dim, num_resblks):
        layers = []
        for i in range(num_resblks): 
            layers.append(ResNetBlock(in_dim=in_dim[i], hidden_dim=hidden_dim[i],))
        return nn.Sequential(*layers)



# audio = torch.randn(1,16,128,80)
# img = torch.randn(1,3,224,224)
# model = AV_SELD()

# # model = Test_demo()
# sed, doa = model(audio, img)
# print('done')