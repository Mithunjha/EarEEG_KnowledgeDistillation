import glob
import numpy as np

def get_list(args,val_data_list):
  psg_sig_list = glob.glob(f'{args.data_path}/x*.h5')
  psg_sig_list.sort()
  val_psg_list = split_data_eval(psg_sig_list,val_data_list)

  label_list = glob.glob(f'{args.data_path}/y*.h5')
  label_list.sort()
  val_label_list = split_data_eval(label_list,val_data_list)
  return val_psg_list,val_label_list

def split_data_eval(data_list,val_list):
    data_list = np.array(data_list)
    val_data_list = data_list[val_list]
    return val_data_list