"""
Dataset Library
"""

import numpy as np
import h5py

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
 

def split_data(data_list,train_list,val_list):
    data_list = np.array(data_list)
    train_data_list = data_list[train_list]
    val_data_list = data_list[val_list]
    return train_data_list, val_data_list



def EEG_process(data,rejected_list,i=0):
    _,_,_,_,_,ELA,ELE,ELI,ERA,ERG,ERE,ERI,_,_,_,_,_,ELB,ELG,ELK,ERB,ERK,_ = data
    reject = rejected_list[i]
    ear_eeg = [ELA,ELE,ELI,ERA,ERG,ERE,ERI,ELB,ELG,ELK,ERB,ERK]
    
    for j in range (len(reject)):
        ear_eeg[j]=ear_eeg[j]*reject[j]
    

    Left_ear = (ear_eeg[0] + ear_eeg[1] + ear_eeg[2] + ear_eeg[7] + ear_eeg[8] + ear_eeg[9])/np.count_nonzero([reject[0],reject[1],reject[2],reject[7],reject[8],reject[9]])   # (ELA + ELE + ELI + ELB + ELG + ELK)/6
    Right_ear = (ear_eeg[3] + ear_eeg[4] + ear_eeg[5] + ear_eeg[6] + ear_eeg[10] + ear_eeg[11])/np.count_nonzero([reject[3],reject[4],reject[5],reject[6],reject[10],reject[11]]) # (ERA + ERG + ERE + ERI + ERB + ERK)/6
    L_R = Left_ear - Right_ear

    if np.count_nonzero([reject[0],reject[7]]) != 0 :
        L_E = (ear_eeg[0]+ear_eeg[7])/np.count_nonzero([reject[0],reject[7]]) - (ear_eeg[1]+ear_eeg[2]+ear_eeg[8]+ear_eeg[9])/np.count_nonzero([reject[1],reject[2],reject[8],reject[9]]) #(ELA + ELB)/2 - (ELE + ELI + ELG + ELK)/4
    else:
        L_E = np.zeros(data.shape[1])
    if np.count_nonzero([reject[3],reject[10]]) != 0 :
        R_E = (ear_eeg[3]+ear_eeg[10])/np.count_nonzero([reject[3],reject[10]]) - (ear_eeg[4]+ear_eeg[5]+ear_eeg[6]+ear_eeg[11])/np.count_nonzero([reject[4],reject[5],reject[6],reject[11]]) # (ERA + ERB)/2 - (ERE + ERI + ERG + ERK)/4
    else:
        R_E = np.zeros(data.shape[1])


    return L_R, L_E, R_E
