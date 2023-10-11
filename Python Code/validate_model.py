from tools.dataset import *
from tools.pointnet_utils import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torchmetrics
from tools.validation import *

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script validates a trained PointNet++ Model
"""

if __name__ == '__main__':

    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # hyperparameters
    num_classes = 5
    batchsize = 6
    thresh = 9     # number of threshes for ROC-Curve
    class_names = ['Ground', 'Building', 'Vegetation', 'Humanmade', 'Bridge']

    # load model
    model = PointNet2(num_classes).to(device)

    # import pretrained model
    model_path = 'pretrained models/model_39_epochs-all_batches-fl-g4_AdamW_lr-scheduler.pth'
    model.load_state_dict(torch.load(model_path))

    # load point cloud
    path = r'D:/Hülsen/Daten/validation'    # path where the validation batches are stored
    val_dataset = PointcloudDataset(path, augmentation=False)   # create Validation Dataset
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=False)

    # initalize validation metrics
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    auroc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    ROC = torchmetrics.ROC(task='multiclass', num_classes=num_classes, thresholds=thresh).to(device)

    # per class validation
    acc_perclass = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average=None).to(device)
    iou_perclass = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes, average=None).to(device)
    auroc_perclass = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average=None).to(device)
    f1_perclass = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)

    # initialize variable for sums
    sum_acc = 0
    sum_iou = 0
    sum_auroc = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    sum_ROC = 0
    sum_fpr = torch.zeros(size=(num_classes, thresh)).to(device)
    sum_tpr = torch.zeros(size=(num_classes, thresh)).to(device)

    counter = torch.zeros(num_classes).to(device)

    sum_acc_perclass = torch.zeros(num_classes).to(device)
    sum_iou_perclass = torch.zeros(num_classes).to(device)
    sum_auroc_perclass = torch.zeros(num_classes).to(device)
    sum_f1_perclass = torch.zeros(num_classes).to(device)
    conf_mat = np.zeros(shape=(num_classes, num_classes))

    # iterate throw batches
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            print(f'epoch {i} from {len(val_dataloader)}')
            # print(f'batch {i + 1} from {len(val_dataloader)}')
            # change labels in to a numeric ordered representation
            labels = labels.to(device)
            inputs = inputs.to(device)
            labels[labels == 2] = 0     # ground
            labels[labels == 6] = 1     # building
            labels[labels == 20] = 2    # vegetation
            labels[labels == 25] = 3    # humanmade
            labels[labels == 30] = 4    # bridge

            # forward pass
            outputs = model(inputs).to(device)

            # get class probabilities
            preds = torch.reshape(F.softmax(outputs, dim=2), shape=(-1, 5))

            # get predicted labels
            _, predictions = torch.max(torch.reshape(outputs, (-1, 5)), 1)

            # add calculated metrics to sum
            sum_acc += acc(predictions.to(device), labels.view(-1)).item()
            sum_iou += iou(predictions.to(device), labels.view(-1)).item()
            sum_precision += precision(predictions.to(device), labels.view(-1)).item()
            sum_recall += recall(predictions.to(device), labels.view(-1)).item()
            sum_f1 += f1(predictions.to(device), labels.view(-1)).item()
            sum_auroc += auroc(preds, labels.view(-1)).item()
            fpr, tpr, thresh = ROC(preds, labels.view(-1).to(torch.int64))

            # if there are no ground truth values for a class ROC will return zeros.
            # we need to count the Non-Zero Rows in order to calculate
            counter += (tpr != 0).any(dim=1)
            sum_fpr += fpr
            sum_tpr += tpr

            # add results to sum to calculate a mean
            sum_acc_perclass += acc_perclass(predictions.to(device), labels.view(-1))
            sum_iou_perclass += iou_perclass(predictions.to(device), labels.view(-1))
            sum_f1_perclass += f1_perclass(predictions.to(device), labels.view(-1))
            sum_auroc_perclass += auroc_perclass(preds, labels.view(-1))

            # calculate a confusion matrix.
            # The tensors have to be on the cpu in order to transform them to a numpy array
            conf_mat += confusion_matrix(labels.view(-1).to(torch.device('cpu')), predictions.to(torch.device('cpu')),
                                         labels=[0, 1, 2, 3, 4], normalize=None)

            del inputs, labels, outputs, preds, predictions, fpr, tpr, thresh

    # get total number of batches
    num_batches = len(val_dataloader)
    # calculate mean metrics
    mean_acc = sum_acc / num_batches
    mean_iou = sum_iou / num_batches
    mean_precision = sum_precision / num_batches
    mean_recall = sum_recall / num_batches
    mean_f1 = sum_f1 / num_batches
    mean_auroc = sum_auroc / num_batches

    mean_fpr = sum_fpr / num_batches
    mean_tpr = sum_tpr / counter.unsqueeze(1)

    # calculate per class metrics
    mean_acc_perclass = sum_acc_perclass / counter
    mean_iou_perclass = sum_iou_perclass / counter
    mean_f1_perclass = sum_f1_perclass / counter
    mean_auroc_perclass = sum_auroc_perclass / counter

    # print results
    result = (f'Validation Accuracy:\t{mean_acc*100:.2f}%\n'
              f'Validation IoU:\t{mean_iou:.4f}\n'
              f'Validation Precision:\t{mean_precision:.3f}\n'
              f'Validation Recall:\t{mean_recall:.3f}\n'
              f'Validation F1:\t{mean_f1:.3f}\n'
              f'Validation Area under ROC:\t{mean_auroc:.3f}\n')

    # save results to a text file
    with open(f'validation/validate_{model_path.split("/")[-1].replace(".pth", "")}.txt', 'w') as f:
        f.write(result)
        f.write('\nPer Class Metrics:\n')
        header = 'Metric\t'
        str_acc = 'Accuracy\t'
        str_iou = 'IoU\t'
        str_f1 = 'F1-Score\t'
        str_auroc = 'Area under ROC\t'

        for i in range(num_classes):
            header += f'{class_names[i]}\t'
            str_acc += f'{(mean_acc_perclass[i].item()):.3f}\t'
            str_iou += f'{(mean_iou_perclass[i].item()):.3f}\t'
            str_f1 += f'{(mean_f1_perclass[i].item()):.3f}\t'
            str_auroc += f'{(mean_auroc_perclass[i].item()):.3f}\t'

        f.write(header + '\n')
        f.write(str_acc + '\n')
        f.write(str_f1 + '\n')
        f.write(str_auroc + '\n')
        f.close()

    print(result)

    # define path, where the ROC- and confusion matrix plot will be stored
    ex_path_roc = f'validation/ROC_{model_path.split("/")[-1].replace(".pth", "")}.png'
    ex_path_confmat = f'validation/confmat_{model_path.split("/")[-1].replace(".pth", "")}.png'

    # Export the ROC-Curves to a defined path
    ROC_Plot(mean_fpr.to(torch.device('cpu')), mean_tpr.to(torch.device('cpu')), num_classes, class_names, ex_path_roc, show=False)
    plt.clf()

    # Export the confusion matrix to a defined path
    plot_confusion_matrix(conf_mat, class_names, ex_path_confmat, show=False)
