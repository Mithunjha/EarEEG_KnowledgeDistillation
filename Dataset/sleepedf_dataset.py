"""
Dataset Library
"""
import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from torch.utils.data import Dataset
from scipy import signal


def read_h5py(path):
  with h5py.File(path, "r") as f:
    print(f"Reading from {path} ====================================================")
    print("Keys in the h5py file : %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data1 = np.array((f[a_group_key]))
    print(f"Number of samples : {len(data1)}")
    print(f"Shape of each data : {data1.shape}")
    return data1
 


def extract_stft(sig, fs, samples_per_seg, overlap, class_label, vis = False):
  f, t, Zxx = signal.stft(sig, fs, nperseg = samples_per_seg, noverlap = overlap, nfft = 256, padded=False)

  #Time wise sum and normalization
  # time_sum_abs = np.reshape(np.sum(np.abs(Zxx[1:-32,1:-1]),axis = 0),(1,Zxx[1:-24,1:-1].shape[1]) )
  # out_abs_Zxx = np.abs(Zxx[1:-32,1:-1])/(time_sum_abs)

  #Epoch wise sum and normalization
  sum_abs = np.sum(np.abs(Zxx[1:-32,0:-1]))
  #print(sum_abs)
  out_abs_Zxx = np.abs(Zxx[1:-32,0:-1])/(sum_abs)

  # print(np.sum(out_abs_Zxx),np.mean(out_abs_Zxx))
  # print(out_Zxx.shape)
  if vis == True:
    plt.figure()  
    plt.pcolormesh(t[0:-1], f[1:-32], np.log(out_abs_Zxx), shading='gouraud',cmap='jet')#,vmin=0, vmax=4) # vmin=0, vmax=amp,
    plt.title(f'Log  STFT Magnitude of class {class_label}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()
    plt.figure()
    plt.pcolormesh(t[0:-1], f[1:-32], out_abs_Zxx, shading='gouraud',cmap='jet') # vmin=0, vmax=amp, /np.sum(np.abs(Zxx[:,1:-1]))
    plt.title(f'STFT Magnitude of class {class_label}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()
    plt.show()
  return f[1:-32],t[0:-1], np.log(out_abs_Zxx)

def extract_stft_multiple_epochs(sig, fs, samples_per_seg, overlap, class_label, vis = False):
  for i in range(sig.shape[0]):
    f,t,Zxx = extract_stft(sig[i].squeeze(), fs, samples_per_seg, overlap, class_label, vis)
    if i == 0:
      Zxx_array = np.reshape(Zxx,(1,Zxx.shape[0],Zxx.shape[1]))
    else:
      Zxx_array = np.concatenate((Zxx_array,np.reshape(Zxx,(1,Zxx.shape[0],Zxx.shape[1]))),axis = 0)
  return f,t,Zxx_array


class SleepEDFDataset(Dataset):
    def __init__(self, data_file_list, label_file_list, device, mean_l = None, sd_l = None, transform=None, target_transform=None, sub_wise_norm = False):
        """
      
        """
        # Get the data
        for i in range(len(data_file_list)):
          if i == 0:
            self.data1 = read_h5py(data_file_list[i])
            self.labels = read_h5py(label_file_list[i])
          else:
            self.data1 = np.concatenate((self.data1, read_h5py(data_file_list[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file_list[i])),axis = 0)
        self.labels = torch.from_numpy(self.labels)

        print(f"Shape of Data : {self.data1.shape} , Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_l)):
            if i == 0:
              self.mean = read_h5py(mean_l[i])
              self.sd = read_h5py(sd_l[i])
            else:
              self.mean = np.concatenate((self.mean, read_h5py(mean_l[i])),axis = 0)
              self.sd = np.concatenate((self.sd, read_h5py(sd_l[i])),axis = 0)
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
        data = self.data1[idx]               
        # print(data.shape)
        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
          data = (data - self.mean[idx]) / self.sd[idx]
        elif self.mean and self.sd:
          data = (data-self.mean)/self.sd

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label



class SleepEDFDataset_STFT(Dataset):
    def __init__(self, data_file_list, label_file_list, device, mean_l = None, sd_l = None, transform=None, target_transform=None, vis = False, sub_wise_norm = False):
        # Get the data
        for i in range(len(data_file_list)):
          if i == 0:
            self.data1 = read_h5py(data_file_list[i])
            self.labels = read_h5py(label_file_list[i])
          else:
            self.data1 = np.concatenate((self.data1, read_h5py(data_file_list[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file_list[i])),axis = 0)
        self.labels = torch.from_numpy(self.labels)

        print(f"Shape of Data : {self.data1.shape} , Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_l)):
            if i == 0:
              self.mean = read_h5py(mean_l[i])
              self.sd = read_h5py(sd_l[i])
            else:
              self.mean = np.concatenate((self.mean, read_h5py(mean_l[i])),axis = 0)
              self.sd = np.concatenate((self.sd, read_h5py(sd_l[i])),axis = 0)
        else:        
          self.mean = mean_l
          self.sd = sd_l
          print(f"Mean : {self.mean} and SD {self.sd}")  
        
        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.vis = vis

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data1[idx]               
        # print(data.shape)
        label = self.labels[idx,]
        if self.sub_wise_norm ==True:
          data = (data - self.mean[idx]) / self.sd[idx]
        elif self.mean and self.sd:
          data = (data-self.mean)/self.sd

        f,t,Zxx = extract_stft(data.squeeze(), fs=100, samples_per_seg=200, overlap=150, class_label=label, vis = self.vis)
        # Zxx = Zxx[1:-24,:]
        # print(Zxx.shape)
        if self.transform:
            Zxx = self.transform(Zxx)
        if self.target_transform:
            label = self.target_transform(label)
        return Zxx, label,f,t
  

class SleepEDFDataset_Sequence(Dataset):
    def __init__(self, data_file_list, label_file_list,sub_file_list, device, num_sequence = 5,
                 mean_l = None, sd_l = None, transform=None, target_transform=None, sub_wise_norm = False):
        """
      
        """
        # Get the data
        for i in range(len(data_file_list)):
          if i == 0:
            self.data1 = read_h5py(data_file_list[i])
            self.labels = read_h5py(label_file_list[i])
            self.sub_file = read_h5py(sub_file_list[i])
          else:
            self.data1 = np.concatenate((self.data1, read_h5py(data_file_list[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file_list[i])),axis = 0)
            self.sub_file = np.concatenate((self.sub_file, read_h5py(sub_file_list[i])),axis = 0)
        self.labels = torch.from_numpy(self.labels)

        print(f"Shape of Data : {self.data1.shape} , Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_l)):
            if i == 0:
              self.mean = read_h5py(mean_l[i])
              self.sd = read_h5py(sd_l[i])
            else:
              self.mean = np.concatenate((self.mean, read_h5py(mean_l[i])),axis = 0)
              self.sd = np.concatenate((self.sd, read_h5py(sd_l[i])),axis = 0)
          self.mean = np.reshape(self.mean ,(self.mean .shape[0],1))
          self.sd = np.reshape(self.sd ,(self.sd.shape[0],1))
        else:        
          
          self.mean = mean_l
          self.sd = sd_l
          print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_sequence = num_sequence

    def __len__(self):
        self.length = len(self.labels)
        return len(self.labels)-self.num_sequence-1

    def __getitem__(self, idx):
      if idx+self.num_sequence <= len(self.labels)-1:
        # print(idx)
        if self.sub_file[idx] == self.sub_file[idx+self.num_sequence-1]:
          sub_idx = self.sub_file[idx:idx+self.num_sequence]
          data = self.data1[idx:idx+self.num_sequence].squeeze()               
          # print(data.shape)
          label = self.labels[idx:idx+self.num_sequence,]
          # print(label.shape)
          if self.sub_wise_norm ==True:
            # print(self.mean.shape)
            data = (data - self.mean[idx:idx+self.num_sequence]) / self.sd[idx:idx+self.num_sequence]
          elif self.mean and self.sd:
            data = (data-self.mean)/self.sd

          if self.transform:
              data = self.transform(data)
          if self.target_transform:
              label = self.target_transform(label)
          return data, label, sub_idx
        
        else:
          # print(idx-self.num_sequence)
          # print("Ignored sequence")
          sub_idx = self.sub_file[idx-self.num_sequence:idx]
          data = self.data1[idx-self.num_sequence:idx].squeeze()               
          # print(data.shape)
          label = self.labels[idx-self.num_sequence:idx,]
          # print(label.shape)
          if self.sub_wise_norm ==True:
            # print(self.mean.shape)
            data = (data - self.mean[idx-self.num_sequence:idx]) / self.sd[idx-self.num_sequence:idx]
          elif self.mean and self.sd:
            data = (data-self.mean)/self.sd

          if self.transform:
              data = self.transform(data)
          if self.target_transform:
              label = self.target_transform(label)
          return data, label, sub_idx
          
          

      else:
        # print(idx-self.num_sequence)
        # print("Over length")
        sub_idx = self.sub_file[idx-self.num_sequence:idx]
        data = self.data1[idx-self.num_sequence:idx].squeeze()               
        # print(data.shape)
        label = self.labels[idx-self.num_sequence:idx,]
        # print(label.shape)
        if self.sub_wise_norm ==True:
          # print(self.mean.shape)
          data = (data - self.mean[idx-self.num_sequence:idx]) / self.sd[idx-self.num_sequence:idx]
        elif self.mean and self.sd:
          data = (data-self.mean)/self.sd

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label, sub_idx



class SleepEDFDataset_Sequence_STFT(Dataset):
    def __init__(self, data_file_list, label_file_list,sub_file_list, device, num_sequence = 5,
                 mean_l = None, sd_l = None, transform=None, target_transform=None, sub_wise_norm = False,vis = False):
        """
      
        """
        # Get the data
        for i in range(len(data_file_list)):
          if i == 0:
            self.data1 = read_h5py(data_file_list[i])
            self.labels = read_h5py(label_file_list[i])
            self.sub_file = read_h5py(sub_file_list[i])
          else:
            self.data1 = np.concatenate((self.data1, read_h5py(data_file_list[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file_list[i])),axis = 0)
            self.sub_file = np.concatenate((self.sub_file, read_h5py(sub_file_list[i])),axis = 0)
        self.labels = torch.from_numpy(self.labels)

        print(f"Shape of Data : {self.data1.shape} , Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_l)):
            if i == 0:
              self.mean = read_h5py(mean_l[i])
              self.sd = read_h5py(sd_l[i])
            else:
              self.mean = np.concatenate((self.mean, read_h5py(mean_l[i])),axis = 0)
              self.sd = np.concatenate((self.sd, read_h5py(sd_l[i])),axis = 0)
          self.mean = np.reshape(self.mean ,(self.mean .shape[0],1))
          self.sd = np.reshape(self.sd ,(self.sd.shape[0],1))
        else:        
          
          self.mean = mean_l
          self.sd = sd_l
          print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_sequence = num_sequence
        self.vis = vis

    def __len__(self):
        self.length = len(self.labels)
        return len(self.labels)-self.num_sequence-1

    def __getitem__(self, idx):
      if idx+self.num_sequence <= len(self.labels)-1:
        # print(idx)
        if self.sub_file[idx] == self.sub_file[idx+self.num_sequence-1]:
          sub_idx = self.sub_file[idx:idx+self.num_sequence]
          data = self.data1[idx:idx+self.num_sequence].squeeze()               
          # print(data.shape)
          label = self.labels[idx:idx+self.num_sequence,]
          # print(label.shape)
          if self.sub_wise_norm ==True:
            # print(self.mean.shape)
            data = (data - self.mean[idx:idx+self.num_sequence]) / self.sd[idx:idx+self.num_sequence]
          elif self.mean and self.sd:
            data = (data-self.mean)/self.sd

          f,t,Zxx = extract_stft_multiple_epochs(data.squeeze(), fs=100, samples_per_seg=200, overlap=150, class_label=label, vis = self.vis)
          Zxx = np.moveaxis(Zxx,0,-1)
          if self.transform:
              Zxx = self.transform(Zxx)
          if self.target_transform:
              label = self.target_transform(label)
          return Zxx, label,f,t
        
        else:
          # print(idx-self.num_sequence)
          # print("Ignored sequence")
          sub_idx = self.sub_file[idx-self.num_sequence:idx]
          data = self.data1[idx-self.num_sequence:idx].squeeze()               
          # print(data.shape)
          label = self.labels[idx-self.num_sequence:idx,]
          # print(label.shape)
          if self.sub_wise_norm ==True:
            # print(self.mean.shape)
            data = (data - self.mean[idx-self.num_sequence:idx]) / self.sd[idx-self.num_sequence:idx]
          elif self.mean and self.sd:
            data = (data-self.mean)/self.sd

          f,t,Zxx = extract_stft_multiple_epochs(data.squeeze(), fs=100, samples_per_seg=200, overlap=150, class_label=label, vis = self.vis)
          Zxx = np.moveaxis(Zxx,0,-1)
          if self.transform:
              Zxx = self.transform(Zxx)
          if self.target_transform:
              label = self.target_transform(label)
          return Zxx, label,f,t
          
          

      else:
        # print(idx-self.num_sequence)
        # print("Over length")
        sub_idx = self.sub_file[idx-self.num_sequence:idx]
        data = self.data1[idx-self.num_sequence:idx].squeeze()               
        # print(data.shape)
        label = self.labels[idx-self.num_sequence:idx,]
        # print(label.shape)
        if self.sub_wise_norm ==True:
          # print(self.mean.shape)
          data = (data - self.mean[idx-self.num_sequence:idx]) / self.sd[idx-self.num_sequence:idx]
        elif self.mean and self.sd:
          data = (data-self.mean)/self.sd

        f,t,Zxx = extract_stft_multiple_epochs(data.squeeze(), fs=100, samples_per_seg=200, overlap=150, class_label=label, vis = self.vis)
        Zxx = np.moveaxis(Zxx,0,-1)
        if self.transform:
            Zxx = self.transform(Zxx)
        if self.target_transform:
            label = self.target_transform(label)
        return Zxx, label,f,t


#############Multi Modal Dataset Class #############################################################

class SleepEDF_MultiChan_Dataset(Dataset):
    def __init__(self, eeg_file, eog_file, emg_file, label_file, device, mean_eeg_l = None, sd_eeg_l = None, 
                 mean_eog_l = None, sd_eog_l = None, mean_emg_l = None, sd_emg_l = None,transform=None, 
                 target_transform=None, sub_wise_norm = False):
        """
      
        """
        # Get the data
        for i in range(len(eeg_file)):
          if i == 0:
            self.eeg = read_h5py(eeg_file[i])
            self.eog = read_h5py(eog_file[i])
            self.emg = read_h5py(emg_file[i])

            self.labels = read_h5py(label_file[i])
          else:
            self.eeg = np.concatenate((self.eeg, read_h5py(eeg_file[i])),axis = 0)
            self.eog = np.concatenate((self.eog, read_h5py(eog_file[i])),axis = 0)
            self.emg = np.concatenate((self.emg, read_h5py(emg_file[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)

        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}, EMG: {self.emg.shape}")
        print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_eeg_l)):
            if i == 0:
              self.mean_eeg  = read_h5py(mean_eeg_l[i])
              self.sd_eeg = read_h5py(sd_eeg_l[i])
              self.mean_eog  = read_h5py(mean_eog_l[i])
              self.sd_eog = read_h5py(sd_eog_l[i])
              self.mean_emg  = read_h5py(mean_emg_l[i])
              self.sd_emg = read_h5py(sd_emg_l[i])
            else:
              self.mean_eeg = np.concatenate((self.mean_eeg, read_h5py(mean_eeg_l[i])),axis = 0)
              self.sd_eeg = np.concatenate((self.sd_eeg, read_h5py(sd_eeg_l[i])),axis = 0)
              self.mean_eog = np.concatenate((self.mean_eog, read_h5py(mean_eog_l[i])),axis = 0)
              self.sd_eog = np.concatenate((self.sd_eog, read_h5py(sd_eog_l[i])),axis = 0)
              self.mean_emg = np.concatenate((self.mean_emg, read_h5py(mean_emg_l[i])),axis = 0)
              self.sd_emg = np.concatenate((self.sd_emg, read_h5py(sd_emg_l[i])),axis = 0)
          print(f"Shapes of Mean  : EEG: {self.mean_eeg.shape}, EOG : {self.mean_eog.shape}, EMG : {self.mean_emg.shape}")
          print(f"Shapes of Sd  : EEG: {self.sd_eeg.shape}, EOG : {self.sd_eog.shape}, EMG : {self.sd_emg.shape}")
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
        eeg_data = self.eeg[idx]   
        eog_data = self.eog[idx]
        emg_data = self.emg[idx]            
        # print(data.shape)
        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
          eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
          eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
          emg_data = (emg_data - self.mean_emg[idx]) / self.sd_emg[idx]
        elif self.mean and self.sd:
          eeg_data = (eeg_data-self.mean[0])/self.sd[0]
          eog_data = (eog_data-self.mean[1])/self.sd[1]
          emg_data = (emg_data-self.mean[2])/self.sd[2]
        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
            emg_data = self.transform(emg_data)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg_data, eog_data, emg_data, label	