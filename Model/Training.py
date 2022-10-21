import torch
from torch import optim as optim
import torch.optim as optim
import time
import warnings
import neptune.new as neptune
 
from Utils.avg_meter import AverageMeter
from Utils.Metrics import accuracy,kappa,confusion_matrix,g_mean


# Training the supervised model
def supervised_training(Net, train_data_loader,val_data_loader, criterion, optimizer, args, device):
    warnings.filterwarnings("ignore")

    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "Supervised training", 
        'Loss': "Categorical Crossentropy Loss",
        'Model Type' : args.model,
        'd_model' : args.d_model,
        'depth' : args.depth,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net

    best_val_acc = 0.0
    best_val_kappa = 0.0
    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        Net.train()
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
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

            if args.model == 'CMT':
                outputs,_,_ = Net(eeg.float().to(device), eog.float().to(device), emg.float().to(device))
            elif args.model == 'USleep':
                outputs,_ = Net(psg.float().to(device))

            loss = criterion(outputs.cpu(), labels)#.to(device))
            loss.backward()
            optimizer.step()
        
            
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

            if args.is_neptune:
                run['train/epoch/batch_loss'].log(losses.val)   
                run['train/epoch/batch_accuracy'].log(train_accuracy.val)
                run['epoch'].log(epoch_idx)

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
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
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

                if args.model == 'CMT':
                    pred,_,_= Net(val_eeg.float().to(device), val_eog.float().to(device), val_emg.float().to(device))
                if args.model == 'USleep':
                    pred,_ = Net(val_psg.float().to(device))
                
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

        

            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
            print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")
        
            print("Validation Results : ")
            print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
            print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")
        

            print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
            print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
            print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")

            if args.is_neptune:
                run['train/epoch/epoch_train_loss'].log(losses.avg)
                run['train/epoch/epoch_val_loss'].log(val_losses.avg)

                run['train/epoch/epoch_train_accuracy'].log(train_accuracy.avg)
                run['train/epoch/epoch_val_accuracy'].log(val_accuracy.avg)

                run['train/epoch/epoch_train_sensitivity'].log(train_sensitivity.avg)
                run['train/epoch/epoch_val_sensitivity'].log(val_sensitivity.avg)

                run['train/epoch/epoch_train_specificity'].log(train_specificity.avg)
                run['train/epoch/epoch_val_specificity'].log(val_specificity.avg)

                run['train/epoch/epoch_train_G-Mean'].log(train_gmean.avg)
                run['train/epoch/epoch_val_G-Mean'].log(val_gmean.avg)

                run['train/epoch/epoch_train_Kappa'].log(train_kappa.avg)
                run['train/epoch/epoch_val_Kappa'].log(val_kappa.avg)

                run['train/epoch/epoch_train_MF1 Score'].log(train_f1_score.avg)
                run['train/epoch/epoch_val_MF1 Score'].log(val_f1_score.avg)

                run['train/epoch/epoch_train_Precision'].log(train_precision.avg)
                run['train/epoch/epoch_val_Precision'].log(val_precision.avg)

      #################################
      
                run['train/epoch/epoch_val_Class wise sensitivity W'].log(class1_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S1'].log(class2_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S2'].log(class3_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S3'].log(class4_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity R'].log(class5_sens.avg)

                run['train/epoch/epoch_val_Class wise specificity W'].log(class1_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S1'].log(class2_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S2'].log(class3_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S3'].log(class4_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity R'].log(class5_spec.avg)

                run['train/epoch/epoch_val_Class wise F1 Score W'].log(class1_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S1'].log(class2_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S2'].log(class3_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S3'].log(class4_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score R'].log(class5_f1.avg)

            if val_accuracy.avg > best_val_acc or (epoch_idx+1)% 50==0 or val_kappa.avg > best_val_kappa:
                if val_accuracy.avg > best_val_acc:
                
                    best_val_acc = val_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_epoch_best_acc.pth.tar')

                if val_kappa.avg > best_val_kappa:
                
                    best_val_kappa = val_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_epoch_best_kappa.pth.tar')

                if (epoch_idx+1)% 50==0 :
                    torch.save(Net, f'{args.project_path}/model_check_points//checkpoint_model_epoch_last.pth.tar')

                if args.is_neptune:
                    run['model/best_acc'].log(val_accuracy.avg)
                    run['model/best_kappa'].log(val_kappa.avg)
    print('========================================Finished Training ===========================================')

    


def KD_online_training(Net_s,Net_t,train_data_loader,val_data_loader,criterion_ce,criterion_mse,optimizer_t, optimizer_s,args,device):
    # Training the model
    warnings.filterwarnings("ignore")
    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "KD online training",
        'Loss': "Categorical Crossentropy + MSE Loss",
        'Model Type' : args.model,
        'depth' : args.depth,
        'd_model' : args.d_model,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net_t

    best_val_acc = 0
    best_val_kappa = 0
    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        
        Net_s.train()
        Net_t.train()        
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses_t = AverageMeter()
        losses_s = AverageMeter()
        val_t_losses = AverageMeter()
        val_s_losses = AverageMeter()
        
        train_t_accuracy = AverageMeter()
        train_s_accuracy = AverageMeter()
        val_t_accuracy = AverageMeter()
        val_s_accuracy = AverageMeter()

        train_t_kappa = AverageMeter()
        train_s_kappa = AverageMeter()
        val_t_kappa = AverageMeter()
        val_s_kappa = AverageMeter()

        end = time.time()

        for batch_idx, data_input in enumerate(train_data_loader):
            
            data_time.update(time.time() - end)
            psg, labels = data_input
            sig1 = psg[:,:,0,:]# L-R
            sig2 = psg[:,:,1,:]# L
            sig3 = psg[:,:,2,:]# R
            sig4 = psg[:,:,3,:]# a1-a2
            sig5 = psg[:,:,4,:]# c3-01
            sig6 = psg[:,:,5,:]# c4-o2
            cur_batch_size = len(sig1)


            if args.model == 'CMT':
                targets,cls_t_feat,_ = Net_t(sig4.float().to(device), sig5.float().to(device), sig6.float().to(device),finetune = True)
                outputs,cls_s_feat,_ = Net_s(sig1.float().to(device), sig2.float().to(device), sig3.float().to(device),finetune = True)
            if args.model == 'USleep':
                targets,cls_t_feat = Net_t(psg[:,:,3:6,:].float().to(device))
                outputs,cls_s_feat = Net_s(psg[:,:,0:3,:].float().to(device))

            loss_t =  criterion_ce(targets.cpu(),labels) 
            loss_s =  criterion_ce(outputs.cpu(),labels) + criterion_mse(cls_s_feat.cpu(), cls_t_feat.detach().cpu())

            optimizer_t.zero_grad()
            loss_t.backward()
            optimizer_t.step()
            
            optimizer_s.zero_grad()
            loss_s.backward()
            optimizer_s.step()

            losses_t.update(loss_t.data.item())
            losses_s.update(loss_s.data.item())

            train_t_accuracy.update(accuracy(targets.cpu(), labels))
            train_s_accuracy.update(accuracy(outputs.cpu(), labels))

            train_t_kappa.update(kappa(targets.cpu(), labels))
            train_s_kappa.update(kappa(outputs.cpu(), labels))
            # print(outputs.shape, labels.shape)
            if args.is_neptune:
                run['train/epoch/batch_t_loss'].log(losses_t.val)     #1
                run['train/epoch/batch_s_loss'].log(losses_s.val) 
                run['train/epoch/batch_t_accuracy'].log(train_t_accuracy.val)
                run['train/epoch/batch_s_accuracy'].log(train_s_accuracy.val)
                run['epoch'].log(epoch_idx)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % 100 == 0:
                
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                    'Train_t_Loss {loss_t.val:.5f} ({loss_t.avg:.5f})\t'\
                    'Train_s_Loss {loss_s.val:.5f} ({loss_s.avg:.5f})\t'\
                    'Train_t_Acc {train_t_acc.val:.5f} ({train_t_acc.avg:.5f})\t'\
                    'Train_s_Acc {train_s_acc.val:.5f} ({train_s_acc.avg:.5f})\t'\
                    'Train_t_Kappa {train_t_kap.val:.5f}({train_t_kap.avg:.5f})\t'\
                    'Train_s_Kappa {train_s_kap.val:.5f}({train_s_kap.avg:.5f})\t'\
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                        speed=data_input[0].size(0)/batch_time.val,
                        data_time=data_time, loss_t=losses_t, loss_s=losses_s, train_t_acc = train_t_accuracy, train_s_acc = train_s_accuracy,
                        train_t_kap = train_t_kappa,train_s_kap = train_s_kappa)
                print(msg)


        #evaluation
        with torch.no_grad():
            Net_s.eval()
            Net_t.eval()
            for batch_val_idx, data_val in enumerate(val_data_loader):
                val_psg, val_labels = data_val
                val_sig1 = val_psg[:,:,0,:]
                val_sig2 = val_psg[:,:,1,:]
                val_sig3 = val_psg[:,:,2,:]
                cur_val_batch_size = len(val_sig1)
                
                if args.model == 'CMT':
                    val_targets,cls_val_t,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                    pred,cls_val_s,_ = Net_s(val_sig1.float().to(device), val_sig2.float().to(device), val_sig3.float().to(device),finetune = True)
                if args.model == 'USleep':
                    val_targets,cls_val_t = Net_t(val_psg[:,:,3:6,:].float().to(device))
                    pred,cls_val_s = Net_s(val_psg[:,:,0:3,:].float().to(device))

                val_t_loss = criterion_ce(val_targets.cpu(),val_labels)
                val_s_loss = criterion_ce(pred.cpu(),val_labels) + criterion_mse(cls_val_s.cpu(), cls_val_t.detach().cpu())

                val_t_losses.update(val_t_loss.data.item())
                val_s_losses.update(val_s_loss.data.item())
                val_t_accuracy.update(accuracy(val_targets.cpu(), val_labels))
                val_s_accuracy.update(accuracy(pred.cpu(), val_labels))

                val_t_kappa.update(kappa(val_targets.cpu(), val_labels))
                val_s_kappa.update(kappa(pred.cpu(), val_labels))

            

            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training T Loss     : {losses_t.avg}, Training T Accuracy      : {train_t_accuracy.avg}, Training T Kappa      : {train_t_kappa.avg}")
            print(f"Training S Loss     : {losses_s.avg}, Training S Accuracy      : {train_s_accuracy.avg}, Training S Kappa      : {train_s_kappa.avg}")
            print("Validation Results : ")
            print(f"Validation T Loss   : {val_t_losses.avg}, Validation T Accuracy : {val_t_accuracy.avg}, Validation T Kappa     : {val_t_kappa.avg}")
            print(f"Validation S Loss   : {val_s_losses.avg}, Validation S Accuracy : {val_s_accuracy.avg}, Validation S Kappa     : {val_s_kappa.avg}")

        
            if args.is_neptune:
                run['train/epoch/epoch_train_t_loss'].log(losses_t.avg)
                run['train/epoch/epoch_train_s_loss'].log(losses_s.avg)
                run['train/epoch/epoch_val_t_loss'].log(val_t_losses.avg)
                run['train/epoch/epoch_val_s_loss'].log(val_s_losses.avg)

                run['train/epoch/epoch_train_t_accuracy'].log(train_t_accuracy.avg)
                run['train/epoch/epoch_train_s_accuracy'].log(train_s_accuracy.avg)
                run['train/epoch/epoch_val_t_accuracy'].log(val_t_accuracy.avg)
                run['train/epoch/epoch_val_s_accuracy'].log(val_s_accuracy.avg)

                run['train/epoch/epoch_train_t_Kappa'].log(train_t_kappa.avg)
                run['train/epoch/epoch_train_s_Kappa'].log(train_s_kappa.avg)
                run['train/epoch/epoch_val_t_Kappa'].log(val_t_kappa.avg)
                run['train/epoch/epoch_val_s_Kappa'].log(val_s_kappa.avg)

            #if val_accuracy.avg > best_val_acc or (epoch_idx+1)%10==0 or val_kappa.avg > best_val_kappa:
            if val_s_accuracy.avg > best_val_acc or val_s_kappa.avg > best_val_kappa:
                if val_s_accuracy.avg > best_val_acc:
                    if args.is_neptune:
                        run['model/bestmodel_acc'].log(epoch_idx+1)
                    best_val_acc = val_s_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net_t, f'{args.project_path}/model_check_points/teacher_checkpoint_acc.pth.tar')
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_acc.pth.tar')

                if val_s_kappa.avg > best_val_kappa:
                    if args.is_neptune:
                        run['model/bestmodel_kappa'].log(epoch_idx+1)
                    best_val_kappa = val_s_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net_t, f'{args.project_path}/model_check_points/teacher_checkpoint_kappa.pth.tar')
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_kappa.pth.tar')

                if args.is_neptune:
                    run['model/best_acc'].log(val_s_accuracy.avg)
                    run['model/best_kappa'].log(val_s_kappa.avg)

    print('========================================Finished Training ===========================================')


def KD_offline_training(Net_s,Net_t,train_data_loader,val_data_loader,criterion_ce,criterion_mse,optimizer_s,args,device):
    # Training the model
    warnings.filterwarnings("ignore")
    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "KD Offline training", 
        'Loss': "Categorical Crossentropy + MSE Loss",
        'Model Type' : args.model,
        'depth' : args.depth,
        'd_model' : args.d_model,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net_t
    best_val_acc = 0
    best_val_kappa = 0

    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        
        Net_s.train()
        Net_t.eval()        ### Check whether weights of the teacher gets updated
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses = AverageMeter()
        val_losses = AverageMeter()
        val_kd_loss = AverageMeter()
        
        train_accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        val_teach_accuracy = AverageMeter()

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
            sig1 = psg[:,:,0,:]# L-R
            sig2 = psg[:,:,1,:]# L
            sig3 = psg[:,:,2,:]# R
            sig4 = psg[:,:,3,:]# c3-01
            sig5 = psg[:,:,4,:]# c4-o2
            sig6 = psg[:,:,5,:]# eog
            cur_batch_size = len(sig1)
            
            # zero the parameter gradients
            

            with torch.no_grad():
                if args.model == 'CMT':
                    targets,cls_t_feat,_ = Net_t(sig4.float().to(device), sig5.float().to(device), sig6.float().to(device),finetune = True)
                if args.model == 'USleep':
                    targets,cls_t_feat = Net_t(psg[:,:,3:6,:].float().to(device))
            optimizer_s.zero_grad()

            if args.model == 'CMT':
                outputs,cls_s_feat,_ = Net_s(sig1.float().to(device), sig2.float().to(device), sig3.float().to(device),finetune = True)
            if args.model == 'USleep':
                outputs,cls_s_feat = Net_s(psg[:,:,0:3,:].float().to(device))

            
            loss =  criterion_ce(outputs.cpu(),labels) + criterion_mse(cls_s_feat,cls_t_feat.detach())
            

            loss.backward()
            optimizer_s.step()
            losses.update(loss.data.item())
            train_accuracy.update(accuracy(outputs.cpu(), labels))

            _,_,_,_,sens,spec,f1, prec = confusion_matrix(outputs.cpu(), labels, 5, cur_batch_size)
            train_sensitivity.update(sens)
            train_specificity.update(spec)
            train_f1_score.update(f1)
            train_precision.update(prec)
            train_gmean.update(g_mean(sens, spec))
            train_kappa.update(kappa(outputs.cpu(), labels))
            # print(outputs.shape, labels.shape)

            if args.is_neptune:
                run['train/epoch/batch_loss'].log(losses.val)     #1
                run['train/epoch/batch_accuracy'].log(train_accuracy.val)
                run['epoch'].log(epoch_idx)
            
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
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                        speed=data_input[0].size(0)/batch_time.val,
                        data_time=data_time, loss=losses, train_acc = train_accuracy,
                        train_sens =train_sensitivity, train_spec = train_specificity, train_gmean = train_gmean,
                        train_kap = train_kappa, train_mf1 = train_f1_score, train_prec = train_precision)
                print(msg)


        #evaluation
        with torch.no_grad():
            Net_s.eval()
            Net_t.eval()
            for batch_val_idx, data_val in enumerate(val_data_loader):
                val_psg, val_labels = data_val
                val_sig1 = val_psg[:,:,0,:]
                val_sig2 = val_psg[:,:,1,:]
                val_sig3 = val_psg[:,:,2,:]
                cur_val_batch_size = len(val_sig1)

                if args.model == 'CMT':
                    val_targets,cls_val_t,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                    pred,cls_val_s,_ = Net_s(val_sig1.float().to(device), val_sig2.float().to(device), val_sig3.float().to(device),finetune = True)
                if args.model == 'USleep':
                    val_targets,cls_val_t = Net_t(val_psg[:,:,3:6,:].float().to(device))
                    pred,cls_val_s = Net_s(val_psg[:,:,0:3,:].float().to(device))


                val_loss = criterion_ce(pred.cpu(), val_labels) + + criterion_mse(cls_val_s,cls_val_t.detach())
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

                val_targets,_,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                val_teach_accuracy.update(accuracy(val_targets.cpu(), val_labels))
                val_kd_loss.update(criterion_mse(pred.cpu(),val_targets.cpu()).data.item())
            print(batch_val_idx)

            

            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
            print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")
            
            print("Validation Results : ")
            print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
            print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")
            print(f"Validation T Acc     : {val_teach_accuracy.avg}, Val_KD_Loss :{val_kd_loss.avg}")

            print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
            print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
            print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")

            if args.is_neptune:
                run['train/epoch/epoch_train_loss'].log(losses.avg)
                run['train/epoch/epoch_val_loss'].log(val_losses.avg)
                run['train/epoch/epoch_val_kd_loss'].log(val_kd_loss.avg)

                run['train/epoch/epoch_train_accuracy'].log(train_accuracy.avg)
                run['train/epoch/epoch_val_accuracy'].log(val_accuracy.avg)
                run['train/epoch/epoch_val_teach_accuracy'].log(val_teach_accuracy.avg)

                run['train/epoch/epoch_train_sensitivity'].log(train_sensitivity.avg)
                run['train/epoch/epoch_val_sensitivity'].log(val_sensitivity.avg)

                run['train/epoch/epoch_train_specificity'].log(train_specificity.avg)
                run['train/epoch/epoch_val_specificity'].log(val_specificity.avg)

                run['train/epoch/epoch_train_G-Mean'].log(train_gmean.avg)
                run['train/epoch/epoch_val_G-Mean'].log(val_gmean.avg)

                run['train/epoch/epoch_train_Kappa'].log(train_kappa.avg)
                run['train/epoch/epoch_val_Kappa'].log(val_kappa.avg)

                run['train/epoch/epoch_train_MF1 Score'].log(train_f1_score.avg)
                run['train/epoch/epoch_val_MF1 Score'].log(val_f1_score.avg)

                run['train/epoch/epoch_train_Precision'].log(train_precision.avg)
                run['train/epoch/epoch_val_Precision'].log(val_precision.avg)

                #################################
                
                run['train/epoch/epoch_val_Class wise sensitivity W'].log(class1_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S1'].log(class2_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S2'].log(class3_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S3'].log(class4_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity R'].log(class5_sens.avg)

                run['train/epoch/epoch_val_Class wise specificity W'].log(class1_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S1'].log(class2_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S2'].log(class3_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S3'].log(class4_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity R'].log(class5_spec.avg)

                run['train/epoch/epoch_val_Class wise F1 Score W'].log(class1_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S1'].log(class2_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S2'].log(class3_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S3'].log(class4_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score R'].log(class5_f1.avg)

            #if val_accuracy.avg > best_val_acc or (epoch_idx+1)%10==0 or val_kappa.avg > best_val_kappa:
            if val_accuracy.avg > best_val_acc or val_kappa.avg > best_val_kappa:
                if val_accuracy.avg > best_val_acc:
                    if args.is_neptune:
                        run['model/bestmodel_acc'].log(epoch_idx+1)
                    best_val_acc = val_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_acc.pth.tar')

                if val_kappa.avg > best_val_kappa:
                    if args.is_neptune:
                        run['model/bestmodel_kappa'].log(epoch_idx+1)
                    best_val_kappa = val_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_kappa.pth.tar')
                if args.is_neptune:
                    run['model/best_acc'].log(val_accuracy.avg)
                    run['model/best_kappa'].log(val_kappa.avg)
              
    print('========================================Finished Training ===========================================')