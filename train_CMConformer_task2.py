from typing import Any, List
import sys

from models.CMConformer import AV_SELD, Test_demo
from custom_dataset import CustomAudioVisualDataset, convert
from metrics import location_sensitive_detection
from utility_functions import gen_submission_list_task2


from torchvision import transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


import pickle

import warnings
warnings.filterwarnings("ignore")



class AudioVisualDatasetModule(pl.LightningDataModule):
    def __init__(self, audio_predictors: List,
                       audio_target: List,
                       image_path=None,
                       image_audio_csv_path=None,
                       transform_image=None,
                       batch_size=16,):
        super().__init__()
        self.audio_predictors = audio_predictors
        self.audio_target = audio_target
        self.image_path = image_path
        self.batch_size = batch_size

        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_csv_path = image_audio_csv_path
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.image_path:
            self.train_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[0], 
                    audio_target=self.audio_target[0], 
                    image_path=self.image_path, 
                    image_audio_csv_path=self.image_audio_csv_path[0], 
                    transform_image=self.transform)

            self.val_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[1], 
                    audio_target=self.audio_target[1], 
                    image_path=self.image_path, 
                    image_audio_csv_path=self.image_audio_csv_path[0], 
                    transform_image=self.transform)

            self.test_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[2], 
                    audio_target=self.audio_target[2], 
                    image_path=self.image_path, 
                    image_audio_csv_path=self.image_audio_csv_path[1], 
                    transform_image=self.transform)
        else:
            self.train_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[0], 
                    audio_target=self.audio_target[0])
            self.val_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[1], 
                    audio_target=self.audio_target[1])
            self.test_set = CustomAudioVisualDataset(audio_predictors=self.audio_predictors[2],
                    audio_target=self.audio_target[2])
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

def create_data(batch_size=2,):

    print("Creating dataset module")

    # Load the pickle data
    audio_preditctors = []
    audio_targets = []
    csv = ['/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/L3DAS23_Task2_train/audio_image.csv','/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/L3DAS23_Task2_dev/audio_image.csv']

    # Train set
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_predictors_train.pkl', 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for train set")
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_target_train.pkl', 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for train set")
    predictor, target = convert(audio_predictor, audio_target)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    # Val set
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_predictors_validation.pkl', 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for val set")
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_target_validation.pkl', 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for val set")
    predictor, target = convert(audio_predictor, audio_target)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    # Test set
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_predictors_test.pkl', 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for test set")
    with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/processed/task2_target_test.pkl', 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for test set")
    predictor, target = convert(audio_predictor, audio_target)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # 重新调整图像大小为 (224, 224)
        transforms.ToTensor(),
    ])
    # Create the dataset module
    dataset_module = AudioVisualDatasetModule(audio_predictors=audio_preditctors, audio_target=audio_targets, 
                            image_path='/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/L3DAS23_Task2_images', 
                            image_audio_csv_path=csv, 
                            transform_image=transform,
                            batch_size=batch_size,)

    print("Created dataset module")
    return dataset_module

def seld_loss(sed, doa, target, criterion_sed, criterion_doa, output_classes=14, class_overlaps=3):
    '''
    compute seld loss as weighted sum of sed (BCE) and doa (MSE) losses
    '''

    #divide labels into sed and doa  (which are joint from the preprocessing)
    target_sed = target[:,:,:output_classes*class_overlaps]
    target_doa = target[:,:,output_classes*class_overlaps:]

    #compute loss
    sed = torch.flatten(sed, start_dim=1)
    doa = torch.flatten(doa, start_dim=1)

    target_sed = torch.flatten(target_sed, start_dim=1)
    target_doa = torch.flatten(target_doa, start_dim=1)

    loss_sed = criterion_sed(sed, target_sed)
    loss_doa = criterion_doa(doa, target_doa)

    return (loss_sed + loss_doa)


def evaluation(sed, doa, target, output_classes=14, class_overlaps=3):
    r"""
        Compute the metrics for the given sed and doa predictions and the target in one step
    """

    #in the target matrices sed and doa are joint
    #divide labels into sed and doa  (which are joint from the preprocessing)
    target_sed = target[:,:,:output_classes*class_overlaps]
    target_doa = target[:,:,output_classes*class_overlaps:]

    # Computing Metrics in one step
    TP = 0
    FP = 0
    FN = 0
    count = 0

    # 需要解析batch中的每个样本
    for i in range(sed.shape[0]):
        sed_ = sed[i,:,:]
        doa_ = doa[i,:,:]
        sed_target = target_sed[i,:,:]
        doa_target = target_doa[i,:,:]

        tp, fp, fn, = compute_metrics(sed_, doa_, sed_target, doa_target, output_classes=14, class_overlaps=3)

        TP += tp
        FP += fp
        FN += fn
    return TP, FP, FN
        


def compute_metrics(sed, doa, sed_target, doa_target, output_classes=14, class_overlaps=3):

    prediction = gen_submission_list_task2(sed, doa, max_overlaps=class_overlaps, max_loc_value=360)

    truth = gen_submission_list_task2(sed_target, doa_target, max_overlaps=class_overlaps, max_loc_value=360)

    tp, fp, fn, _ = location_sensitive_detection(prediction, truth, 300,
                                                      1.75, False) # 300 is max num frames, 1.75 is the spatial threshold (ref to evaluate_baseline_task2.py)

    return tp, fp, fn

def compute_f_score(TP, FP, FN, epsilon=sys.float_info.epsilon):
    r"""
        Compute the F score for the given TP, FP and FN
    """
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    F_score = 2 * ((precision * recall) / (precision + recall + epsilon))
    return precision, recall, F_score


class Model(pl.LightningModule):
    def __init__(self, input_channels=16, n_bins=128, num_classes=14, num_resblks=4, num_confblks=2, hidden_dim=512, kernel_size=3, num_heads=4, dropout=0.1, output_classes=14, class_overlaps=3,
                    lr=0.05, warmup_epochs=10,
                    #lr=1e-4, weight_decay=1e-6, max_epochs=100, batch_size=16, num_workers=4, gpus=1, precision=16, amp_level='O1', deterministic=True, seed=42
                    ):
        super().__init__()
        
        self.model = AV_SELD(input_channels=input_channels, n_bins=n_bins, num_classes=num_classes, num_resblks=num_resblks, num_confblks=num_confblks, hidden_dim=hidden_dim, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout, output_classes=output_classes, class_overlaps=class_overlaps)
        # self.model = Test_demo(input_channels=input_channels, n_bins=n_bins, num_classes=num_classes, num_resblks=num_resblks, num_confblks=num_confblks, hidden_dim=hidden_dim, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout, output_classes=output_classes, class_overlaps=class_overlaps)
        

        self.criterion_sed = nn.BCELoss()
        self.criterion_doa = nn.MSELoss()

        self.num_epochs_warmup = warmup_epochs
        self.lr = lr

        # results recording dict
        # # self.train_ = {'loss': [], 'TP': [], 'FP': [], 'FN': []}     #, 'precision': [], 'recall': [], 'F_score': []}
        # self.val_ = {'loss': [], 'TP': [], 'FP': [], 'FN': []}       #, 'precision': [], 'recall': [], 'F_score': []}
        # self.test_ = {'loss': [], 'TP': [], 'FP': [], 'FN': []}      #,  'precision': [], 'recall': [], 'F_score': []}

        self.val_TP = []
        self.val_FP = []
        self.val_FN = []

        self.test_TP = []
        self.test_FP = []
        self.test_FN = []
        



    def forward(self, audio, img):
        sed, doa = self.model(audio, img)
        return sed, doa

    def training_step(self, batch, batch_idx):
        (audio, img), target = batch
        sed, doa = self.model(audio, img)
        train_loss = seld_loss(sed, doa, target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)
        # tp, fp, fn = compute_metrics(sed, doa, target, output_classes=14, class_overlaps=3)
        

        # # record into result dict
        # self.train_['loss'].append(train_loss)
        # self.train_['TP'].append(tp)
        # self.train_['FP'].append(fp)
        # self.train_['FN'].append(fn)

        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': train_loss}

    # def training_epoch_end(self):

    #     # compute one epoch's metrics
    #     TP = sum(self.train_['TP'])
    #     FP = sum(self.train_['FP'])
    #     FN = sum(self.train_['FN'])
    #     # compute one epoch's F score
    #     precision, recall, F_score = compute_f_score(TP, FP, FN)

    #     # record into 
    #     self.log('train_f_score', F_score, logger=True)
    #     self.log('train_precision', precision, logger=True)
    #     self.log('train_recall', recall, logger=True)

    #     # free the memory
    #     self.train_.clear()


    def validation_step(self, batch, batch_idx):
        (audio, img), target = batch
        sed, doa = self.model(audio, img)
        val_loss = seld_loss(sed, doa, target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)
        tp, fp, fn = evaluation(sed.cpu(), doa.cpu(), target.cpu(), output_classes=14, class_overlaps=3)

        # record into result dict
        # self.val_['loss'].append(val_loss)
        self.val_TP.append(tp)
        self.val_FP.append(fp)
        self.val_FN.append(fn)

        self.log('val_loss', val_loss, logger=True)
        return {'val_loss': val_loss}

    def on_validation_epoch_end(self):
         
        # compute one epoch's metrics
        TP = sum(self.val_TP)
        FP = sum(self.val_FP)
        FN = sum(self.val_FN)
        # compute one epoch's F score
        precision, recall, F_score = compute_f_score(TP, FP, FN)
    
        # record into 
        self.log('val_f_score', F_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
        # free the memory
        self.val_TP.clear()
        self.val_FP.clear()
        self.val_FN.clear()

    def test_step(self, batch, batch_idx):
        audio, img = batch
        sed, doa = self.model(audio, img)
        test_loss = seld_loss(sed.cpu(), doa.cpu(), target.cpu(), self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)
        tp, fp, fn = evaluation(sed, doa, target, output_classes=14, class_overlaps=3)

        # record into result dict
        # self.test_['loss'].append(test_loss)
        # self.test_['TP'].append(tp)
        # self.test_['FP'].append(fp)
        # self.test_['FN'].append(fn)

        self.test_TP.append(tp)
        self.test_FP.append(fp)
        self.test_FN.append(fn)


        self.log('test_loss', test_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'test_loss': test_loss}

    def on_test_epoch_end(self):
             
        # compute one epoch's metrics
        # TP = sum(self.test_['TP'])
        # FP = sum(self.test_['FP'])
        # FN = sum(self.test_['FN'])
        TP = sum(self.test_TP)
        FP = sum(self.test_FP)
        FN = sum(self.test_FN)
        # compute one epoch's F score
        precision, recall, F_score = compute_f_score(TP, FP, FN)
        
        # record into 
        self.log('test_f_score', F_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_precision', precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recall', recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # free the memory
        # self.test_.clear()
        self.test_TP.clear()
        self.test_FP.clear()
        self.test_FN.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True) #, weight_decay=1e-6) ref to train_baseline_task2.py
        
        # lr_lambda = lambda epoch: self.lr * np.minimum(
        #     (epoch + 1) ** -0.5, (epoch + 1) * (self.num_epochs_warmup ** -1.5)
        # )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0) # ref to train_baseline_task2.py
        return [optimizer]#, [scheduler]



### MAIN ###

# data_module = create_data(batch_size=16,)
# data_module.setup()

# (audio,img), target = data_module.train_set.__getitem__(0)
# print(audio.shape)
# print(img.shape)
# print(target.shape)

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("medium")

wandb_logger = WandbLogger(project='CMConformer', 
                            entity='', 
                            log_model=True, 
                            save_dir='/mnt/fast/nobackup/scratch4weeks/pw00391/wandb_log')


seed_everything(10, workers=True)

data_module = create_data(batch_size=64)
model = Model(lr=0.00001)
# model = Model.load_from_checkpoint(checkpoint_path='/mnt/fast/nobackup/scratch4weeks/pw00391/ckpt/epoch=49-val_loss=0.12-val_f_score=0.00.ckpt', lr=0.00001, map_location=torch.device('cpu'))

EarlyStopping = EarlyStopping(monitor='val_f_score', patience=10, mode='max')
checkpoint_callback = ModelCheckpoint(dirpath='/mnt/fast/nobackup/scratch4weeks/pw00391/ckpt', filename='{epoch}-{val_loss:.2f}-{val_f_score:.2f}')

trainer = Trainer(fast_dev_run=False,  
                  min_epochs=50, 
                  max_epochs=400,
                  accelerator='auto',  
                  deterministic=True,
                  enable_checkpointing=True, 
                  callbacks=[EarlyStopping, checkpoint_callback], 
                  logger=wandb_logger
                  )

trainer.fit(model, data_module)
# trainer.test(model, data_module.test_dataloader())

