import warnings
import torch
import argparse
import numpy as np
from torch.utils import data
from torchvision import transforms


from Datasets.EarEEG_utils import read_h5py
from Utils.Metrics import accuracy,kappa,confusion_matrix,g_mean
from Utils.eval_utils import get_list,split_data_eval
from Datasets.Dataset_loader import EarEEG_MultiChan_Dataset

def parse_option():
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--model_path', type=str, default="",   help='Path to saved checkpoint for retraining')
    parser.add_argument('--batch_size', type=int, default = 32 ,  help='Batch Size')
    parser.add_argument('--val_data_list', nargs="+", default = [8] ,  help='Folds in the dataset for validation')
    parser.add_argument('--signals', type=str, default = 'ear-eeg'  ,choices=['ear-eeg', 'scalp-eeg'],  help='signal type')
    parser.add_argument('--data_path', type=str, default='',help='Path to the dataset file')
    opt = parser.parse_args()
    return opt


def main():
    warnings.filterwarnings("ignore")
    args = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    first_main = 0
    test_model = torch.load(f'{args.model_path}')
    print(f"Test Model Loaded :  {args.model_path}")

    val_psg_list,val_label_list = get_list(args,args.val_data_list)


    rejection_list = read_h5py(f"{args.data_path}/rejected.h5")
    
    val_reject_list = split_data_eval(rejection_list,args.val_data_list)

    val_dataset = EarEEG_MultiChan_Dataset(psg_file = val_psg_list, 
                                          label_file = val_label_list, args = args, 
                                          device = device, 
                                          reject_list = val_reject_list,
                                          sub_wise_norm = False,
                                          transform=transforms.Compose([
                                            transforms.ToTensor(),]) )

    val_data_loader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)
  
    test_model.eval()   
    
        
    print(f"Get Predictions For Subject {args.val_data_list[0]+1} =====================================================>")
    labels_val_main = []
    pred_val_main = []
    first = 0 
    with torch.no_grad():
        test_model.eval()
        for batch_val_idx, data_val in enumerate(val_data_loader):
            psg, val_labels = data_val
            sig1 = psg[:,:,0,:]# L-R
            sig2 = psg[:,:,1,:]# L
            sig3 = psg[:,:,2,:]# R

            pred,_,_ = test_model(sig1.float().to(device), sig2.float().to(device), sig3.float().to(device),finetune = True)

            if first == 0:
                labels_val_main = val_labels.cpu().numpy()
                pred_val_main = pred.cpu().numpy()
                first = 1
            else:
                labels_val_main = np.concatenate((labels_val_main, val_labels.cpu().numpy()))
                pred_val_main =  np.concatenate((pred_val_main,pred.cpu().numpy()))
    
    
    print(f"Evaluations For Subject {args.val_data_list[0]+1} =====================================================>")
    sens_l,spec_l,f1_l,prec_l, sens,spec,f1,prec = confusion_matrix(torch.from_numpy(pred_val_main), torch.from_numpy(labels_val_main),5, labels_val_main.shape[0], print_conf_mat=True)
    g = g_mean(sens, spec)
    acc = accuracy(torch.from_numpy(pred_val_main), torch.from_numpy(labels_val_main))
    kap = kappa(torch.from_numpy(pred_val_main), torch.from_numpy(labels_val_main))
    print(f"Val Set:{args.val_data_list[0]+1}")
    print(f"Accuracy {acc}")
    print(f"Kappa {kap}")
    print(f"Macro F1 Score {f1}")
    print(f"G Mean {g}")
    print(f"Sensitivity {sens}")
    print(f"Specificity {spec}")
    print(f"Class wise F1 Score {f1_l}")

    if first_main ==0:
        first_main = 1
        main_all_labels = labels_val_main
        main_all_pred = pred_val_main
    else:
        main_all_labels = np.concatenate((main_all_labels, labels_val_main))
        main_all_pred = np.concatenate((main_all_pred,pred_val_main))

    print(main_all_labels.shape)
    print(main_all_pred.shape)


if __name__ == '__main__':
    main()