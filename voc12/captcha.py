import os
import torch
import pickle
import numpy as np

class Normalize():
  def __init__(self,mean,std):
    self.mean=mean
    self.std=std
  def __call__(self, sample):
    sample["data"]-=self.mean
    sample["data"]/=self.std
    return sample

class MakeItStupid():
  def __call__(self, sample):
    sample["data"]=0*sample["data"]
    if sample["gt"]==1: sample["data"][12:20,12:20]=255
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

class CAPTCHADataLoader():
  def __init__(self,root_dir,batch_size,transform=None):
    self.ids=[file.split(".npy")[0] for file in os.listdir(root_dir)]
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
        "data": self.data[slice_id], "gt": 1 if "vessel" in self.id else 0
    }
    if self.transform:
      sample = self.transform(sample)
    return sample

class AllBrainImages(torch.utils.data.Dataset):
  def __init__(self,root_dir,transform=None):
    self.dataloader=CAPTCHADataLoader(root_dir,1,transform=transform)
    self.map=[j for i,brain in enumerate(self.dataloader) for j in [i]*len(brain)]

  def __len__(self):
    return sum(len(brain) for brain in self.dataloader)
  
  def __getitem__(self, id):
    brain_id=self.map[id]
    slice_id=id-self.map.index(brain_id)
    return self.dataloader.loaders[brain_id].dataset.__getitem__(slice_id)