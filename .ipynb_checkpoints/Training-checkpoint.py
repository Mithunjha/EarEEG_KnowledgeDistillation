import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
from pathlib import Path
from torch.utils import data
import math
from PIL import Image
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch.utils.data import Dataset, DataLoader
import time
import glob

 
from Datasets.EarEEG_utils import split_data, EEG_process, read_h5py
from Datasets.Dataset import EarEEG_MultiChan_Dataset
from Model.CrossModalTransformer import Cross_Transformer_Network


###########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

experiment = 1
######################

train_data_list = [1,2,3,5,6,8,7]  #4 discarded 
val_data_list = [0]

psg_sig_list = glob.glob('mmsm/EarEEG_Dataset/x*.h5')
psg_sig_list.sort()
[train_psg_list, val_psg_list] = split_data(psg_sig_list,train_data_list,val_data_list)

label_list = glob.glob('mmsm/EarEEG_Dataset/y*.h5')
label_list.sort()
[train_label_list, val_label_list] = split_data(label_list,train_data_list,val_data_list)

rejection_list = read_h5py("mmsm/EarEEG_Dataset/rejected.h5")
[train_reject_list, val_reject_list] = split_data(rejection_list,train_data_list,val_data_list)

################################################

train_dataset = EarEEG_MultiChan_Dataset(psg_file = train_psg_list, 
                                       label_file = train_label_list, device = device, 
                                       reject_list = train_reject_list,
                                       sub_wise_norm = False,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),]) )
val_dataset = EarEEG_MultiChan_Dataset(psg_file = val_psg_list, 
                                       label_file = val_label_list, device = device, 
                                       reject_list = val_reject_list,
                                       sub_wise_norm = False,
                                       transform=transforms.Compose([
                                        transforms.ToTensor(),]) )

batch_size = 32
train_data_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
val_data_loader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True)

####################################


d_model = 256 #256 
dim_feedforward=1024  #1024
window_size = 50#25 50
Net = Cross_Transformer_Network(d_model = d_model, dim_feedforward=dim_feedforward,window_size = window_size ).to(device)
# Net =  torch.load('/content/drive/MyDrive/EarEEG/EAREEGV2-112/checkpoint_model_epoch_best_kappa2.pth.tar').to(device)

lr = 0.001#0.001
beta_1 =  0.9    
beta_2 =  0.999    
eps = 1e-9
n_epochs = 20

criterion = nn.CrossEntropyLoss()#weight=weights) #####I didnt use weithgs for Ear EEG
optimizer = torch.optim.Adam(Net.parameters(), lr=lr, betas=(beta_1, beta_2),eps = eps, weight_decay = 0.0001)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) 


# Training the model
import warnings
warnings.filterwarnings("ignore")

best_val_acc = 0.0
best_val_kappa = 0.0
for epoch_idx in range(n_epochs):  # loop over the dataset multiple times
    Net.train()
    print(f'===========================================================Training Epoch : [{epoch_idx+1}/{n_epochs}] ===========================================================================================================>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    losses = AverageMeter()
    val_losses = AverageMeter()
    
    train_accuracy = AverageMeter()
    val_accuracy = AverageMeter()

    train_sensitivity = AverageMeter()
    val_sensitivity = AverageMeter()
    
    train_specificity = AverageMeter()
    val_specificity = AverageMeter()

    train_gmean = AverageMeter()
    val_gmean = AverageMeter()

    train_kappa = AverageMeter()
    val_kappa = AverageMeter()

    train_f1_score = AverageMeter()
    val_f1_score = AverageMeter()

    train_precision = AverageMeter()
    val_precision = AverageMeter()

    class1_sens = AverageMeter()
    class2_sens = AverageMeter()
    class3_sens = AverageMeter()
    class4_sens = AverageMeter()
    class5_sens = AverageMeter()

    class1_spec = AverageMeter()
    class2_spec = AverageMeter()
    class3_spec = AverageMeter()
    class4_spec = AverageMeter()
    class5_spec = AverageMeter()

    class1_f1 = AverageMeter()
    class2_f1 = AverageMeter()
    class3_f1 = AverageMeter()
    class4_f1 = AverageMeter()
    class5_f1 = AverageMeter()

    end = time.time()

    for batch_idx, data_input in enumerate(train_data_loader):
        # get the inputs; data is a list of [inputs, labels]
        data_time.update(time.time() - end)
        psg, labels = data_input
        eeg = psg[:,:,0,:]
        eog = psg[:,:,1,:]
        emg = psg[:,:,2,:]
        cur_batch_size = len(eeg)
        

        optimizer.zero_grad()

    
        outputs,_,_ = Net(eeg.float().to(device), eog.float().to(device), emg.float().to(device))
     
        

        loss = criterion(outputs.cpu(), labels)#.to(device))
        

        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        losses.update(loss.data.item())
        train_accuracy.update(accuracy(outputs.cpu(), labels))

        _,_,_,_,sens,spec,f1, prec = confusion_matrix(outputs.cpu(), labels, 5, cur_batch_size)
        train_sensitivity.update(sens)
        train_specificity.update(spec)
        train_f1_score.update(f1)
        train_precision.update(prec)
        train_gmean.update(g_mean(sens, spec))
        train_kappa.update(kappa(outputs.cpu(), labels))
        
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       

        if batch_idx % 100 == 0:
            
            msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                  'Train_Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                  'Train_Acc {train_acc.val:.5f} ({train_acc.avg:.5f})\t'\
                  'Train_G-Mean {train_gmean.val:.5f}({train_gmean.avg:.5f})\t'\
                  'Train_Kappa {train_kap.val:.5f}({train_kap.avg:.5f})\t'\
                  'Train_MF1 {train_mf1.val:.5f}({train_mf1.avg:.5f})\t'\
                  'Train_Precision {train_prec.val:.5f}({train_prec.avg:.5f})\t'\
                  'Train_Sensitivity {train_sens.val:.5f}({train_sens.avg:.5f})\t'\
                  'Train_Specificity {train_spec.val:.5f}({train_spec.avg:.5f})\t'\
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                      epoch_idx+1, batch_idx, len(train_data_loader),n_epochs, batch_time=batch_time,
                      speed=data_input[0].size(0)/batch_time.val,
                      data_time=data_time, loss=losses, train_acc = train_accuracy,
                      train_sens =train_sensitivity, train_spec = train_specificity, train_gmean = train_gmean,
                      train_kap = train_kappa, train_mf1 = train_f1_score, train_prec = train_precision)
            print(msg)


    #evaluation
    with torch.no_grad():
        Net.eval()
        for batch_val_idx, data_val in enumerate(val_data_loader):
            val_psg, val_labels = data_val
            val_eeg = val_psg[:,:,0,:]
            val_eog = val_psg[:,:,1,:]
            val_emg = val_psg[:,:,2,:]
            cur_val_batch_size = len(val_eeg)
            pred,_,_= Net(val_eeg.float().to(device), val_eog.float().to(device), val_emg.float().to(device))


            val_loss = criterion(pred.cpu(), val_labels)#.to(device))
            val_losses.update(val_loss.data.item())
            val_accuracy.update(accuracy(pred.cpu(), val_labels))

            sens_list,spec_list,f1_list,prec_list, sens,spec,f1,prec = confusion_matrix(pred.cpu(), val_labels,  5, cur_val_batch_size)
            val_sensitivity.update(sens)
            val_specificity.update(spec)
            val_f1_score.update(f1)
            val_precision.update(prec)
            val_gmean.update(g_mean(sens, spec))
            val_kappa.update(kappa(pred.cpu(), val_labels))

            class1_sens.update(sens_list[0])
            class2_sens.update(sens_list[1])
            class3_sens.update(sens_list[2])
            class4_sens.update(sens_list[3])
            class5_sens.update(sens_list[4])

            class1_spec.update(spec_list[0])
            class2_spec.update(spec_list[1])
            class3_spec.update(spec_list[2])
            class4_spec.update(spec_list[3])
            class5_spec.update(spec_list[4])

            class1_f1.update(f1_list[0])
            class2_f1.update(f1_list[1])
            class3_f1.update(f1_list[2])
            class4_f1.update(f1_list[3])
            class5_f1.update(f1_list[4])

        print(batch_val_idx)

     

        print(f'===========================================================Epoch : [{epoch_idx+1}/{n_epochs}]  Evaluation ===========================================================================================================>')
        print("Training Results : ")
        print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
        print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")
      
        print("Validation Results : ")
        print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
        print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")
    

        print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
        print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
        print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")



        if val_accuracy.avg > best_val_acc or (epoch_idx+1)% 50==0 or val_kappa.avg > best_val_kappa:
            if val_accuracy.avg > best_val_acc:
            
                best_val_acc = val_accuracy.avg
                print("================================================================================================")
                print("                                          Saving Best Model (ACC)                                     ")
                print("================================================================================================")
                torch.save(Net, f'mmsm/EarEEG_Dataset/{experiment}/checkpoint_model_epoch_best_acc2.pth.tar')

            if val_kappa.avg > best_val_kappa:
            
                best_val_kappa = val_kappa.avg
                print("================================================================================================")
                print("                                          Saving Best Model (Kappa)                                    ")
                print("================================================================================================")
                torch.save(Net, f'mmsm/EarEEG_Dataset/{experiment}/checkpoint_model_epoch_best_kappa2.pth.tar')

            if (epoch_idx+1)% 50==0 :
                torch.save(Net, f'mmsm/EarEEG_Dataset/{experiment}/checkpoint_model_epoch_last.pth.tar')
         
print('========================================Finished Training ===========================================')

print("finished Training")


