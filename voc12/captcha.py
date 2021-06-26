import os
import random
import torch
import pickle
import numpy as np
import voc12.data

from PIL import Image

class Standardize():
  def __init__(self,mean,std):
    self.mean=mean
    self.std=std
  def __call__(self, sample):
    sample["data"]-=self.mean
    sample["data"]/=self.std
    return sample

class Normalize():
  def __init__(self,minimum,maximum):
    self.minimum=minimum
    self.maximum=maximum
  def __call__(self, sample):
    sample["data"]-=self.minimum
    sample["data"]/=(self.maximum-self.minimum)
    return sample

class MakeItStupid():
  def __call__(self, sample):
    sample["data"]=0*sample["data"]
    if sample["gt"]==1: sample["data"][12:20,12:20]=255
    return sample

class CenterCrop():
  def __call__(self, sample):
    tmp=np.zeros([448,448])
    #tmp[208:240,208:240]=sample["data"]
    tmp[176:272,176:272]=sample["data"]
    sample["data"]=tmp
    return sample

class OneHot():
  def one_hot(self,seg,num_classes=2):#4->2
    return np.eye(num_classes)[seg]
  def __call__(self, sample):
    sample['gt']=self.one_hot(sample['gt'])
    return sample

class ToTensor():
  def __call__(self, sample):
    sample['data']=torch.from_numpy(sample['data'][None,:,:]).float()
    sample['gt']=torch.from_numpy(sample['gt']).float()
    return sample["gt"],sample["data"],sample["gt"]

class RandomResizeLong():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def resize(self, img):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        img = img.resize(target_shape, resample=Image.CUBIC)

        return img
    
    def __call__(self, sample):
        img = Image.fromarray((255*sample["data"]).astype(np.uint8))
        sample["data"]=np.array(self.resize(img),dtype=float)
        return sample
    

class CAPTCHADataLoader():
  def __init__(self,root_dir,batch_size,transform=None):
    self.ids=[file.split(".npy")[0] for file in os.listdir(root_dir)]
    #self.ids=[
    #  image_id for image_id in self.ids if "vessel" in image_id or image_id=="002_32_empy" or image_id=="004_32_empty" or image_id=="005_32_empty"
    #]
    self.batch_size=batch_size
    self.loaders=[]
    for id in self.ids:
      self.loaders.append(torch.utils.data.DataLoader(
          BrainImage(root_dir,id,transform=transform),
          batch_size=batch_size,shuffle=False,num_workers=0
      ))
    self.counter_id=0

  def __iter__(self):
    self.counter_iter=0
    return self

  def set_transform(self,transform):
    for loader in self.patient_loaders:
      loader.dataset.transform=transform
  
  def __next__(self):
    if(self.counter_iter==len(self)):
      raise StopIteration
    loader=self.loaders[self.counter_id]
    self.counter_id+=1
    self.counter_iter+=1
    if(self.counter_id%len(self)==0):
      self.counter_id=0
    return loader

  def __len__(self):
    return len(self.ids)

  def current_id(self):
    return self.ids[self.counter_id]

class BrainImage(torch.utils.data.Dataset):
  def __init__(self,root_dir,id,transform=None):
    self.root_dir=root_dir
    self.id=id
    """with open("preprocessed/patient_info.pkl",'rb') as f:
      self.info=pickle.load(f)[patient_id]"""
    self.data=np.load(os.path.join(self.root_dir,f"{self.id}.npy")).astype(float)
    self.transform=transform

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, slice_id):
    sample={
        #"data": self.data[slice_id], "gt": 1 if "vessel" in self.id else 0
        "data": self.data[slice_id], "gt": 1 if slice_id < 500 else 0
    }
    if self.transform:
      sample = self.transform(sample)
    return sample

class AllBrainImages(torch.utils.data.Dataset):
  def __init__(self,root_dir,transform=None,label_la_dir=None,label_ha_dir=None,label_transform=None):
    self.dataloader=CAPTCHADataLoader(root_dir,1,transform=transform)
    self.map=[j for i,brain in enumerate(self.dataloader) for j in [i]*len(brain)]
    self.label_la_dir=label_la_dir
    self.label_ha_dir=label_ha_dir
    self.label_transform=label_transform
    self.extract_aff_lab_func = voc12.data.ExtractAffinityLabelInRadius(cropsize=96//8)

  def __len__(self):
    return sum(len(brain) for brain in self.dataloader)
  
  def __getitem__(self, id):
    brain_id=self.map[id]
    slice_id=id-self.map.index(brain_id)
    dataset=self.dataloader.loaders[brain_id].dataset
    if self.label_la_dir is not None and self.label_ha_dir is not None:
      label_la_path = os.path.join(self.label_la_dir, dataset.id + '.npy')
      label_ha_path = os.path.join(self.label_ha_dir, dataset.id + '.npy')
      label_la = np.load(label_la_path)#,allow_pickle=True).item()
      label_ha = np.load(label_ha_path)#,allow_pickle=True).item()
      #label = np.array(list(label_la.values()) + list(label_ha.values()))
      label = np.stack([label_la,label_ha])
      label = label[:,slice_id]
      label = np.transpose(label, (1, 2, 0))
      if self.label_transform: label = self.label_transform(label)
      no_score_region = np.max(label, -1) < 1e-5
      label_la, label_ha = np.array_split(label, 2, axis=-1)
      label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
      label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
      label = label_la.copy()
      label[label_la == 0] = 255
      label[label_ha == 0] = 0
      label[no_score_region] = 255 # mostly outer of cropped region
      label = self.extract_aff_lab_func(label)
      return dataset.__getitem__(slice_id),label

    return dataset.__getitem__(slice_id)