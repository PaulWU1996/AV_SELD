from typing import Any, List
import sys

from models.CMConformer import AV_SELD
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
import argparse

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
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

def create_data(args):

    print("Creating dataset module")

    # Load the pickle data
    audio_preditctors = []
    audio_targets = []
    csv = [args.path_csv_images_train, args.path_csv_images_test]

    # Train set
    with open(args.training_predictors_path, 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for train set")
    with open(args.training_target_path, 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for train set")
    predictor, target = convert(audio_predictor, audio_target, chunk_length=args.chunk_lengths)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    # Val set
    with open(args.validation_predictors_path, 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for val set")
    with open(args.validation_target_path, 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for val set")
    predictor, target = convert(audio_predictor, audio_target, chunk_length=args.chunk_lengths)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    # Test set
    with open(args.test_predictors_path, 'rb') as f:
        audio_predictor = pickle.load(f)
        # audio_preditctors.append(audio_predictor)
        print("Loaded audio predictors for test set")
    with open(args.test_target_path, 'rb') as f:
        audio_target = pickle.load(f)
        # audio_targets.append(audio_target)
        print("Loaded audio targets for test set")
    predictor, target = convert(audio_predictor, audio_target, chunk_length=args.chunk_lengths)
    audio_preditctors.append(predictor)
    audio_targets.append(target)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # 重新调整图像大小为 (224, 224)
        transforms.ToTensor(),
    ])
    # Create the dataset module
    dataset_module = AudioVisualDatasetModule(audio_predictors=audio_preditctors, audio_target=audio_targets, 
                            image_path=args.path_images, 
                            image_audio_csv_path=csv, 
                            transform_image=transform,
                            batch_size=args.batch_size,)

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

    return loss_sed #(loss_sed + loss_doa)


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

        # tp, fp, fn, = compute_metrics(sed_, doa_, sed_target, doa_target, output_classes=14, class_overlaps=3)
        tp, fp, fn, = compute_metrics(sed_, doa_target, sed_target, doa_target, output_classes=14, class_overlaps=3)

        TP += tp
        FP += fp
        FN += fn
    return TP, FP, FN
        


def compute_metrics(sed, doa, sed_target, doa_target, output_classes=14, class_overlaps=3):

    # prediction = gen_submission_list_task2(sed, doa, max_overlaps=class_overlaps, max_loc_value=360)
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
    def __init__(self,
                res_in = [8, 64, 128, 256], res_out = [64, 128, 256, 512], 
                n_bins=256, num_resblks=4, num_confblks=2, hidden_dim=512, kernel_size=3, num_heads=4, dropout=0.1,
                audio_visual=True, chunk_lengths=1.0,
                output_classes=14, class_overlaps=3,
                lr=0.00005, warmup_epochs=10,):
        super().__init__()
        
        #self.model = AV_SELD(input_channels=input_channels, n_bins=n_bins, num_classes=num_classes, num_resblks=num_resblks, num_confblks=num_confblks, hidden_dim=hidden_dim, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout, output_classes=output_classes, class_overlaps=class_overlaps)
        self.model = AV_SELD(res_in=res_in, res_out=res_out, n_bins=n_bins, num_resblks=num_resblks, num_confblks=num_confblks, hidden_dim=hidden_dim, kernel_size=kernel_size, num_heads=num_heads, dropout=dropout, audio_visual=audio_visual, chunk_lengths=chunk_lengths, output_classes=output_classes, class_overlaps=class_overlaps)

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
        # sed, doa = self.model(audio, img)
        sed = self.model(audio)
        return sed#, doa

    def training_step(self, batch, batch_idx):
        (audio, img), target = batch
        sed, doa = self.model(audio, img)

        # train_loss = seld_loss(sed, doa, target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)

        train_loss = seld_loss(sed, torch.zeros((sed.shape[0],10,126), device=sed.device), target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)

        self.log('train_loss', train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': train_loss}


    def validation_step(self, batch, batch_idx):
        (audio, img), target = batch
        sed, doa = self.model(audio, img)
        # val_loss = seld_loss(sed, doa, target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)

        # sed = self.model(audio)
        val_loss = seld_loss(sed, torch.zeros((sed.shape[0],10,126),device=sed.device), target, self.criterion_sed, self.criterion_doa, output_classes=14, class_overlaps=3)



        # tp, fp, fn = evaluation(sed.cpu(), doa.cpu(), target.cpu(), output_classes=14, class_overlaps=3)
        tp, fp, fn = evaluation(sed.cpu(), torch.zeros((sed.shape[0],10,126), device='cpu'), target.cpu(), output_classes=14, class_overlaps=3)

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=True) #, weight_decay=1e-6) ref to train_baseline_task2.py
        
        # lr_lambda = lambda epoch: self.lr * np.minimum(
        #     (epoch + 1) ** -0.5, (epoch + 1) * (self.num_epochs_warmup ** -1.5)
        # )
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0) # ref to train_baseline_task2.py
        return [optimizer]#, [scheduler]


def main(args):
    
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("medium")

    # set wandb logger
    wandb_logger = WandbLogger(project='CMConformer', 
                            entity='', 
                            log_model=True, 
                            save_dir=args.log_path)
    
    seed_everything(10, workers=True)
    data_module = create_data(args)

    model = Model(res_in=args.res_in, res_out=args.res_out, n_bins=args.n_bins, num_resblks=args.num_resblks, num_confblks=args.num_confblks, hidden_dim=args.hidden_dim, kernel_size=args.kernel_size, num_heads=args.num_heads, dropout=args.dropout, output_classes=args.output_classes, class_overlaps=args.class_overlaps, lr=args.lr, warmup_epochs=args.warmup_epochs, audio_visual=args.audio_visual, chunk_lengths=args.chunk_lengths)

    earlyStopping = EarlyStopping(monitor='val_f_score', patience=args.patience, mode='max')
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt_path, filename='{epoch}-{val_loss:.2f}-{val_f_score:.2f}')

    trainer = Trainer(fast_dev_run=False,  
                    min_epochs=args.min_epochs, 
                    max_epochs=args.max_epochs,
                    accelerator='auto',  
                    deterministic=True,
                    enable_checkpointing=True, 
                    callbacks=[earlyStopping, checkpoint_callback],
                    # auto_lr_find=True, 
                    logger=wandb_logger
                    )

    # import lightning as L

    # tuner = L.pytorch.tuner.Tuner(trainer)
    # tuner.lr_find(model, data_module)

    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    trainer.fit(model, data_module)
    # trainer.test(model, data_module.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # saving/loading
    parser.add_argument('--ckpt_path', type=str, default='/vol/research/VS-Work/PW00391/icassp/ckpt',
                        help='path to save the checkpoint')
    parser.add_argument('--ckpt_name', type=str, default=None,
                        help='name of the checkpoint to load')
    parser.add_argument('--log_path', type=str, default='/vol/research/VS-Work/PW00391/icassp/log',
                        help='path to save the log')
    parser.add_argument('--path_images', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/L3DAS23_Task2_images',
                        help="Path to the folder containing all images of Task2. None when using the audio-only version")    
    parser.add_argument('--path_csv_images_train', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/L3DAS23_Task2_train/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo) in the train/val set")
    parser.add_argument('--path_csv_images_test', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/L3DAS23_Task2_dev/audio_image.csv',
                        help="Path to the CSV file for the couples (name_audio, name_photo) in the test set")
    
    # dataset parameters
    parser.add_argument('--training_predictors_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_predictors_train.pkl')
    parser.add_argument('--training_target_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_target_train.pkl')
    parser.add_argument('--validation_predictors_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_predictors_validation.pkl')
    parser.add_argument('--validation_target_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_target_validation.pkl')
    parser.add_argument('--test_predictors_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_predictors_test.pkl')
    parser.add_argument('--test_target_path', type=str, default='/vol/research/VS-Work/PW00391/L3DAS23/Output/processed/task2_target_test.pkl')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--min_epochs', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=5)

    # model parameters
    parser.add_argument('--res_in',type=list, default=[4, 64, 128, 256],
                        help="Number of input channels for each residual block")
    parser.add_argument('--res_out',type=list, default=[64, 128, 256, 512],
                        help="Number of output channels for each residual block")
    parser.add_argument('--n_bins', type=int, default=256,
                        help="Number of frequency bins")
    parser.add_argument('--num_resblks', type=int, default=4,
                        help="Number of residual blocks")
    parser.add_argument('--num_confblks', type=int, default=2,
                        help="Number of conformer/cmconformer blocks")
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help="Number of hidden dimensions in conformer/cmconformer blocks")
    parser.add_argument('--kernel_size', type=int, default=3,
                        help="Kernel size for the conformer/cmconformer blocks")
    parser.add_argument('--num_heads', type=int, default=4,
                        help="Number of heads for the conformer/cmconformer blocks")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate for the conformer/cmconformer blocks")
    parser.add_argument('--output_classes', type=int, default=14,
                        help="Number of classes for the sed head")
    parser.add_argument('--class_overlaps', type=int, default=3,
                        help="Number of overlapping frames for each class")
    parser.add_argument('--audio_visual', type=bool, default=True,)
    parser.add_argument('--chunk_lengths', type=float, default=1.0,
                        help="Length of the chunks in seconds")

    args = parser.parse_args()
    
    # debug and test
    args.audio_visual = False #True if args.path_images else False
    # args.max_epochs = 1
    # args.min_epochs = 1
    args.validation_audio_predictors = args.training_predictors_path
    args.validation_audio_target = args.training_target_path
    args.lr = 0.001 # 0.04 AV

    main(args)
    # data_module = create_data(args)
    # print('data_module')



