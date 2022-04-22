
from Datasets.EarEEG_utils import EEG_process, read_h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
import glob

from Datasets.EarEEG_utils import split_data

class EarEEG_MultiChan_Dataset(Dataset):
    def __init__(self, psg_file, label_file,args, device, mean_l = None, sd_l = None,reject_list = None, transform=None, target_transform=None, sub_wise_norm = False):
        """
      # C3,O1,A1,F3,LOC,ELA,ELE,ELI,ERA,ERG,ERE,ERI,C4,O2,A2,F4,ROC,ELB,ELG,ELK,ERB,ERK,CHIN12
        """
        # Get the data
        for i in range(len(psg_file)):
            if i == 0:
                if reject_list.any():
                    psg_signal = read_h5py(psg_file[i])
                    if args.signals == 'ear-eeg':
                      L_R, L_E, R_E = EEG_process(psg_signal,reject_list,i=i)
                      L_R = np.reshape(L_R,(L_R.shape[0],1,L_R.shape[1]))
                      L_E = np.reshape(L_E,(L_E.shape[0],1,L_E.shape[1]))
                      R_E = np.reshape(R_E,(R_E.shape[0],1,R_E.shape[1]))
                      self.psg = np.concatenate((L_R,L_E,R_E),axis = 1) 

                    if args.signals == 'scalp-eeg':
                      eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
                      eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
                      eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))
                      self.psg = np.concatenate((eeg_a1_a2,eeg_c3_o1,eeg_c4_o2),axis = 1)

                    self.labels = read_h5py(label_file[i])
                else:  
                    self.psg = read_h5py(psg_file[i])
                    self.labels = read_h5py(label_file[i])
            else:
                if reject_list.any():
                    psg_signal = read_h5py(psg_file[i])
                    if args.signals == 'ear-eeg':
                      L_R, L_E, R_E = EEG_process(psg_signal,reject_list,i=i)
                      L_R = np.reshape(L_R,(L_R.shape[0],1,L_R.shape[1]))
                      L_E = np.reshape(L_E,(L_E.shape[0],1,L_E.shape[1]))
                      R_E = np.reshape(R_E,(R_E.shape[0],1,R_E.shape[1]))
                      psg_comb = np.concatenate((L_R,L_E,R_E),axis = 1) 

                    if args.signals == 'scalp-eeg':
                      eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
                      eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
                      eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))
                      psg_comb = np.concatenate((eeg_a1_a2,eeg_c3_o1,eeg_c4_o2),axis = 1) 

                    self.psg = np.concatenate((self.psg,psg_comb),axis = 0)
                    self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

                else:
                    self.psg = np.concatenate((self.psg, read_h5py(psg_file[i])),axis = 1)
                    self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)
        print(f"Data shape : {self.psg.shape}")
        print(f"Labels shape : {self.labels.shape}")
        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels/self.labels.shape[0]}")
        print(f"Labels count weights: {1/(bin_labels/self.labels.shape[0])}")

        if sub_wise_norm == True:
            print(f"Reading Subject wise mean and sd")
            for i in range(len(mean_l)):
                if i == 0:
                    self.mean_l  = read_h5py(mean_l[i])
                    self.sd_l = read_h5py(sd_l[i])
                else:
                    self.mean_l = np.concatenate((self.mean_l, read_h5py(mean_l[i])),axis = 1)
                    self.sd_l = np.concatenate((self.sd_l, read_h5py(sd_l[i])),axis = 1)

            print(f"Shapes of Mean  : {self.mean_l.shape}")
            print(f"Shapes of Sd    : {self.sd_l.shape}")
        else:     
            self.mean = mean_l
            self.sd = sd_l
            print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        psg_data = self.psg[idx]         
        # print(data.shape)
        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
            psg_data = (psg_data - self.mean_l[idx]) / self.sd_l[idx]
        elif self.mean and self.sd:
            psg_data = (psg_data - self.mean_l) / self.sd_l

        if self.transform:
            psg_data = self.transform(psg_data)
        if self.target_transform:
            label = self.target_transform(label)
        return psg_data, label


class EarEEG_KD_Dataset(Dataset):
    def __init__(self, psg_file, label_file, device, mean_l = None, sd_l = None,
                reject_list = None, transform=None, target_transform=None, sub_wise_norm = False):
        """
      # C3,O1,A1,F3,LOC,ELA,ELE,ELI,ERA,ERG,ERE,ERI,C4,O2,A2,F4,ROC,ELB,ELG,ELK,ERB,ERK,CHIN12
        """
        # Get the data
        for i in range(len(psg_file)):
          if i == 0:
            if reject_list.any():
              psg_signal = read_h5py(psg_file[i])
              L_R, L_E, R_E = EEG_process(psg_signal,reject_list,i=i)

              L_R = np.reshape(L_R,(L_R.shape[0],1,L_R.shape[1]))
              L_E = np.reshape(L_E,(L_E.shape[0],1,L_E.shape[1]))
              R_E = np.reshape(R_E,(R_E.shape[0],1,R_E.shape[1]))

              eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
              eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
              eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))

 
              self.psg = np.concatenate((L_R,L_E,R_E,eeg_a1_a2,eeg_c3_o1,eeg_c4_o2),axis = 1) 
              self.labels = read_h5py(label_file[i])
            else:  
              self.psg = read_h5py(psg_file[i])
              self.labels = read_h5py(label_file[i])
          else:
            if reject_list.any():
              psg_signal = read_h5py(psg_file[i])
              L_R, L_E, R_E = EEG_process(psg_signal,reject_list,i=i)
              L_R = np.reshape(L_R,(L_R.shape[0],1,L_R.shape[1]))
              L_E = np.reshape(L_E,(L_E.shape[0],1,L_E.shape[1]))
              R_E = np.reshape(R_E,(R_E.shape[0],1,R_E.shape[1]))

   
              eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
              eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
              eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))

              psg_comb = np.concatenate((L_R,L_E,R_E,eeg_a1_a2,eeg_c3_o1,eeg_c4_o2),axis = 1) 

              self.psg = np.concatenate((self.psg,psg_comb),axis = 0)
              self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

            else:
              self.psg = np.concatenate((self.psg, read_h5py(psg_file[i])),axis = 1)
              self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)
        print(f"Data shape : {self.psg.shape}")
        print(f"Labels shape : {self.labels.shape}")
        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels/self.labels.shape[0]}")
        print(f"Labels count weights: {1/(bin_labels/self.labels.shape[0])}")
        
        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_l)):
            if i == 0:
              self.mean_l  = read_h5py(mean_l[i])
              self.sd_l = read_h5py(sd_l[i])
            else:
              self.mean_l = np.concatenate((self.mean_l, read_h5py(mean_l[i])),axis = 1)
              self.sd_l = np.concatenate((self.sd_l, read_h5py(sd_l[i])),axis = 1)

          print(f"Shapes of Mean  : {self.mean_l.shape}")
          print(f"Shapes of Sd    : {self.sd_l.shape}")
        else:     
          self.mean = mean_l
          self.sd = sd_l
          print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        psg_data = self.psg[idx] 
        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
          psg_data = (psg_data - self.mean_l[idx]) / self.sd_l[idx]
        elif self.mean and self.sd:
          psg_data = (psg_data - self.mean_l) / self.sd_l

        if self.transform:
            psg_data = self.transform(psg_data)
        if self.target_transform:
            label = self.target_transform(label)
        return psg_data, label



def get_dataset(args,device):
  psg_sig_list = glob.glob(f'{args.data_path}/x*.h5')
  psg_sig_list.sort()
  [train_psg_list, val_psg_list] = split_data(psg_sig_list,args.train_data_list,args.val_data_list)

  label_list = glob.glob(f'{args.data_path}/y*.h5')
  label_list.sort()
  [train_label_list, val_label_list] = split_data(label_list,args.train_data_list,args.val_data_list)

  rejection_list = read_h5py(f'{args.data_path}/rejected.h5')
  [train_reject_list, val_reject_list] = split_data(rejection_list,args.train_data_list,args.val_data_list)

  ################################################

  if args.training_type == "supervised":

    train_dataset = EarEEG_MultiChan_Dataset(psg_file = train_psg_list, 
                                          label_file = train_label_list, args = args, 
                                          device = device, 
                                          reject_list = train_reject_list,
                                          sub_wise_norm = False,
                                          transform=transforms.Compose([
                                            transforms.ToTensor(),]) )
    val_dataset = EarEEG_MultiChan_Dataset(psg_file = val_psg_list, 
                                          label_file = val_label_list, args = args, 
                                          device = device, 
                                          reject_list = val_reject_list,
                                          sub_wise_norm = False,
                                          transform=transforms.Compose([
                                            transforms.ToTensor(),]) )

  if args.training_type == "Knowledge_distillation":
    train_dataset = EarEEG_KD_Dataset(psg_file = train_psg_list, 
                                       label_file = train_label_list, 
                                       device = device, 
                                       reject_list = train_reject_list,
                                       sub_wise_norm = False,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),]) )
    val_dataset = EarEEG_KD_Dataset(psg_file = val_psg_list, 
                                       label_file = val_label_list, device = device, 
                                       reject_list = val_reject_list,
                                       sub_wise_norm = False,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),]) )

  
  train_data_loader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
  val_data_loader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)

  return train_data_loader,val_data_loader

####################################