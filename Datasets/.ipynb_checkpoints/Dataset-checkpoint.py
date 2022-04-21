
import Datasets.sleepedf_dataset
from Datasets.sleepedf_dataset import read_h5py
from Datasets.EarEEG_data import EEG_process


class EarEEG_MultiChan_Dataset(Dataset):
    def __init__(self, psg_file, label_file, device, mean_l = None, sd_l = None,reject_list = None, transform=None, target_transform=None, sub_wise_norm = False):
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

#                     eeg_f3_o1 = np.reshape(psg_signal[3] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_f4_o2 = np.reshape(psg_signal[15] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_c3_a2 = np.reshape(psg_signal[0] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_c4_a1 = np.reshape(psg_signal[12] - psg_signal[2],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
#                     eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))

#                     eog =  np.reshape(psg_signal[16] - psg_signal[4],(L_R.shape[0],1,L_R.shape[-1]))
#                     emg = np.reshape(psg_signal[22],(L_R.shape[0],1,L_R.shape[-1]))
                
                
                    # self.psg = np.concatenate((L_R,eog,emg),axis = 1) 
                    self.psg = np.concatenate((L_R,L_E,R_E),axis = 1) 
                    # self.psg = np.concatenate((L_E,R_E,eog),axis = 1) 
                    # self.psg = np.concatenate((eeg_a1_a2,eeg_c3_o1,eeg_c4_o2),axis = 1) 
                    # self.psg = np.concatenate((eeg_a1_a2,eeg_c3_a2,eeg_c4_a1),axis = 1) 
                    # self.psg = np.concatenate((eeg_c3_a1,eeg_c4_a2,eog),axis = 1) 

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

#                   eeg_f3_o1 = np.reshape(psg_signal[3] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_f4_o2 = np.reshape(psg_signal[15] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_c3_a2 = np.reshape(psg_signal[0] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_c4_a1 = np.reshape(psg_signal[12] - psg_signal[2],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_c3_o1 = np.reshape(psg_signal[0] - psg_signal[1],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_c4_o2 = np.reshape(psg_signal[12] - psg_signal[13],(L_R.shape[0],1,L_R.shape[-1]))
#                   eeg_a1_a2 = np.reshape(psg_signal[2] - psg_signal[14],(L_R.shape[0],1,L_R.shape[-1]))

#                   eog =  np.reshape(psg_signal[16] - psg_signal[4],(L_R.shape[0],1,L_R.shape[-1]))
#                   emg = np.reshape(psg_signal[22],(L_R.shape[0],1,L_R.shape[-1]))
                
                
                  # psg_comb = np.concatenate((L_R,eog,emg),axis = 1) 
                    psg_comb = np.concatenate((L_R,L_E,R_E),axis = 1) 
                  # psg_comb = np.concatenate((L_E,R_E,eog),axis = 1) 
                  # psg_comb = np.concatenate((eeg_a1_a2,eeg_c3_a2,eeg_c4_a1),axis = 1) 
                  # psg_comb = np.concatenate((eeg_c3_a1,eeg_c4_a2,eog),axis = 1) 


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

