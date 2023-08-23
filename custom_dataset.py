import os
import numpy as np
import torch
import torch.utils.data as utils

from utility_functions import audio_image_csv_to_dict, load_image

# import torchvision.transforms as transforms

# # 定义一些常用的图像转换操作
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),     # 重新调整图像大小为 (224, 224)
#     # transforms.RandomHorizontalFlip(),  # 随机水平翻转
#     transforms.ToTensor(),              # 转换为张量
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
# ])

class CustomAudioVisualDataset(utils.Dataset):
    def __init__(self, audio_predictors, audio_target, image_path=None, image_audio_csv_path=None, transform_image=None):
        self.audio_predictors = audio_predictors[0]
        self.audio_target = audio_target
        self.audio_predictors_path = audio_predictors[1]
        self.image_path = image_path
        if image_path:
            print("AUDIOVISUAL ON")
            self.image_audio_dict = audio_image_csv_to_dict(image_audio_csv_path)
            self.transform = transform_image
        else:
            print("AUDIOVISUAL OFF")
    
    def __len__(self):
        return len(self.audio_predictors)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_pred = self.audio_predictors[idx]
        audio_trg = self.audio_target[idx]
        audio_pred_path = self.audio_predictors_path[idx]
        
        if self.image_path:
            image_name = self.image_audio_dict[audio_pred_path]
            img = load_image(os.path.join(self.image_path, image_name))
            
            if self.transform:
                img = self.transform(img)

            return (audio_pred.astype(np.float32), img.to(torch.float32)), audio_trg.astype(np.float32)
        
        return audio_pred.astype(np.float32), audio_trg.astype(np.float32)
    
# class CustomBatch:
#     def __init__(self, data):
#         transposed_data = list(zip(*data))
#         transposed_data_0 = list(zip(*transposed_data[0]))
#         self.audio_pred = torch.stack(transposed_data_0[0], 0)
#         self.inp = list(zip(self.audio_pred, transposed_data_0[1]))
#         self.tgt = torch.stack(transposed_data[1], 0)

#     # custom memory pinning method on custom type
#     def pin_memory(self):
#         self.audio_pred = self.audio_pred.pin_memory()
#         self.tgt = self.tgt.pin_memory()
#         return self

# def collate_wrapper(batch):
#     return CustomBatch(batch)


def convert(predictors, targets, chunk_length=1):
    r"""
    Input:
        predictors: (list) [audio_predictors, audio_predictors_path]
        targets: (list) [audio_target]
        chunk_length: (int) length of chunk in seconds (default: 1)
    Output:
        out_predictors: (list) [audio_predictors, audio_predictors_path]
        out_targets: (list) [audio_target]
    """

    out_predictors = []
    out_features = []
    out_files = []
    out_targets = []

    label_frame_in_chunk = int(chunk_length * 10)
    feat_frame_in_chunk = int(label_frame_in_chunk * 8)

    nfile = len(predictors[0])
    for fn in range(nfile):
        feature = predictors[0][fn]
        # feature = np.concatenate((feature[:,:128,:],feature[:,128:,:]), axis=0)

        file_name = predictors[1][fn]
        target = targets[fn]
        all_steps = feature.shape[2]

        chunk_step = all_steps // feat_frame_in_chunk
        
        for i in range(chunk_step):
            out_features.append(feature[:,:,i*feat_frame_in_chunk:(i+1)*feat_frame_in_chunk])
            out_files.append(file_name)

            out_targets.append(target[i*label_frame_in_chunk:(i+1)*label_frame_in_chunk])
            
    out_predictors = [out_features, out_files]

    return out_predictors, out_targets


# import pickle
# from torchvision import transforms
# import matplotlib.pyplot as plt

# with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Output/processed/task2_predictors_train.pkl', 'rb') as f:
#     audio_predictors = pickle.load(f)

# with open('/mnt/fast/nobackup/scratch4weeks/pw00391/Output/processed/task2_target_train.pkl', 'rb') as f:
#     audio_target = pickle.load(f)


# predictors, targets = convert(audio_predictors, audio_target)

# transform = transforms.Compose([
#         transforms.Resize((224, 224)),     # 重新调整图像大小为 (224, 224)
#         transforms.ToTensor(),
#     ])

# train_set = CustomAudioVisualDataset(audio_predictors=predictors, 
#                 audio_target=targets, 
#                 image_path='/mnt/fast/nobackup/scratch4weeks/pw00391/Task2/L3DAS23_Task2_images', 
#                 image_audio_csv_path='Task2/L3DAS23_Task2_train/audio_image.csv', 
#                 transform_image=transform)



# print("Train set length: ", len(train_set))
# (audio,img), target = train_set.__getitem__(3)
# # plt.imshow(img.permute(1, 2, 0))
# # %%
# plt.figure()
# plt.imshow(audio[0], cmap='viridis')
# plt.show()
# print('done')

