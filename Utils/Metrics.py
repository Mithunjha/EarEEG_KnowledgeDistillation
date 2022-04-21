from torch.autograd import Variable
from sklearn.metrics import cohen_kappa_score

def accuracy(outputs, labels):
    # m = nn.Softmax(dim=1) #########
    # outputs = m(outputs)   ########
    # print(sum(outputs[0,:]))
    pred = torch.argmax(outputs, 1)
    # print(pred)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    total = int(labels.shape[0])
    return correct / total

def kappa(output, label):  
  # m = nn.Softmax(dim=1) #########
  # output = m(output)   ########
    preds = torch.argmax(output, 1)
  # y_true = y_targets.cpu().numpy()
  # y_pred = y_preds.cpu().numpy()
    return cohen_kappa_score(label, preds)



def g_mean(sensitivity, specificity):
    return (sensitivity*specificity)**0.5



def confusion_matrix(output, label, n_classes, batch_size, print_conf_mat = False):
    # m = nn.Softmax(dim=1) #########
    # output = m(output)   ########
    preds = torch.argmax(output, 1)
    # print(preds)
    # print(label)
    conf_matrix = torch.zeros(n_classes, n_classes)
    avg_sensitivity = 0
    avg_specificity = 0
    avg_F1_score = 0
    avg_precision = 0
    sens_list = []
    spec_list = []
    F1_list = []
    precision_list = []

    for p, t in zip(preds, label):
        conf_matrix[p, t] += 1
    if print_conf_mat==True:    ##Jathu made this edit
        print(conf_matrix)

        plot_confusion_matrix(cm = conf_matrix.cpu().numpy(),
                      normalize    = True,
                      target_names = ['Wake', 'N1', 'N2','N3','REM'],
                      title        = "Confusion Matrix (5-Class)")
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title('Confusion Matrix')
        plt.colorbar()
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.get_cmap('Blues'), alpha=1)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=int(conf_matrix[i, j]), va='center', ha='center', size='xx-large')
 
        plt.xlabel('Prediction', fontsize=18)
        plt.ylabel('Ground Truth', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.colorbar()
        plt.show()

    TP = conf_matrix.diag()
    for c in range(n_classes):
        idx = torch.ones(n_classes).byte()
        idx[c] = 0
        TN = conf_matrix[idx.nonzero()[:,None], idx.nonzero()].sum()
        FP = conf_matrix[c, idx].sum()
        FN = conf_matrix[idx, c].sum()

        if (TP[c]+FN) != 0:
            sensitivity = (TP[c] / (TP[c]+FN))
        else:
            sensitivity = 0

        if (TN+FP) != 0:
            specificity = (TN / (TN+FP))
        else:
            specificity = 0

        if ((2*TP[c]) + (FN + FP)) !=0:
            F1_score = (2*TP[c])/((2*TP[c]) + (FN + FP))
        else:
            F1_score = 0
        
        if (TP[c]+FP) !=0:
            precision = (TP[c]/(TP[c]+FP))
        else:
            precision = 0


        sens_list.append(float(sensitivity))
        spec_list.append(float(specificity))
        F1_list.append(float(F1_score))
        precision_list.append(float(precision))

        avg_sensitivity += float(sensitivity)
        avg_specificity += float(specificity)
        avg_F1_score += float(F1_score)
        avg_precision +=float(precision)

    return sens_list, spec_list,F1_list, precision_list, avg_sensitivity/5, avg_specificity/5, avg_F1_score/5, avg_precision/5 