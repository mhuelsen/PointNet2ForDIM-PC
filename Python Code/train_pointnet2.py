from tools.dataset import *
from tools.pointnet_utils import *

from torch.utils.data import DataLoader
import torchmetrics
from torch.optim.lr_scheduler import ExponentialLR

import logging
import time
from tqdm import tqdm

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script trains a PointNet++ Model
"""


if __name__ == '__main__':

    # create and config a logger
    logging.basicConfig(filename='log/pointnet2.log',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('\n\n\n')

    # path to batchfile
    train_path = r'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/train'
    val_path = r'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/validation'
    logger.info(f'\nTrain path: {train_path}\nValidation path: {val_path}')

    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize Dataset
    train_dataset = PointcloudDataset(train_path, augmentation=True)   # --> Object defined in tools/dataset.py
    val_dataset = PointcloudDataset(val_path, augmentation=False)
    logger.info(f'Imported train and test data')

    # set hyperparameter
    num_epochs = 40
    batch_size = 7
    num_classes = 5
    learning_rate = 0.001

    # initalize metrics for evaluation
    acc_per_class = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device)
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes).to(device)

    # write hyperparameter to logger
    logger.info(f'HYPERPARAMETERS: number of epochs: {num_epochs}, '
                f'batchsize: {batch_size}, '
                f'number of classes: {num_classes}')

    # initialize DataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    logger.info('DataLoader is initialized.\n')

    # define Model
    model = PointNet2(num_classes).to(device)
    # import model if a pretrained model is available
    model.load_state_dict(torch.load('pretrained models/model_39_epochs-all_batches-fl-g4_AdamW_lr-scheduler.pth'))

    # loss and optimizer
    criterion = FocalLoss(gamma=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    # initialize the best validation accuracy
    best_val_acc = 0.9550

    for epoch in range(num_epochs):
        # initialize starting time for process timing
        start_time = time.time()

        sum_loss = 0
        sum_acc = 0
        sum_iou = 0

        logger.info(f'Starting with epoch Nr.{epoch+1}')
        # iterate throw epochs
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False):
            # move input and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # change labels to a numeric order
            labels[labels == 2] = 0     # ground
            labels[labels == 6] = 1     # building
            labels[labels == 20] = 2    # vegetation
            labels[labels == 25] = 3    # humanmade
            labels[labels == 30] = 4    # bridges

            # forward pass
            outputs = model(inputs).to(device)
            # calculate loss
            loss = criterion(torch.reshape(outputs, (-1, 5)), torch.reshape(labels, (-1,)).to(torch.int64))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # add metrics to the sum in order to calculate the mean
                sum_loss += loss.item()
                _, predictions = torch.max(torch.reshape(outputs, (-1, 5)), 1)
                sum_acc += acc(predictions.to(device), labels.view(-1).to(device)).item()
                sum_iou += iou(predictions.to(device), labels.view(-1).to(device)).item()

        # do step with lr-scheduler
        if epoch >= 10:
            scheduler.step()

        # calculate mean metrics
        mean_loss = sum_loss / len(train_dataloader)
        train_acc = sum_acc / len(train_dataloader)
        train_iou = sum_iou / len(train_dataloader)

        # calculate metrics for validation data
        with torch.no_grad():

            end_time = time.time()
            time_elapsed = end_time - start_time

            sum_val_loss = 0
            sum_val_acc = 0
            sum_val_class_acc = np.zeros(5)
            sum_val_iou = 0
            sum_val_f1 = 0

            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels[labels == 2] = 0
                labels[labels == 6] = 1
                labels[labels == 20] = 2
                labels[labels == 25] = 3
                labels[labels == 30] = 4

                # forward pass
                outputs = model(inputs).to(device)
                # get prediction as the max value of the output
                _, predictions = torch.max(torch.reshape(outputs, (-1, 5)), 1)

                sum_val_acc += acc(predictions, labels.view(-1))
                sum_val_loss += criterion(torch.reshape(outputs, (-1, 5)), torch.reshape(labels, (-1,)).to(torch.int64))
                sum_val_iou += iou(predictions, labels.view(-1))
                sum_val_f1 += f1(predictions, labels.view(-1))
                sum_val_class_acc += acc_per_class(predictions, labels.view(-1)).to(torch.device('cpu')).numpy()

            mean_val_loss = (sum_val_loss / len(val_dataloader)).item()
            mean_val_acc = (sum_val_acc / len(val_dataloader)).item()
            mean_val_iou = (sum_val_iou / len(val_dataloader)).item()
            mean_val_f1 = (sum_val_f1 / len(val_dataloader)).item()
            mean_val_class_acc = sum_val_class_acc / len(val_dataloader)

        message = (f'ep {int(epoch + 1):02d}/{int(num_epochs):02d}\tloss={mean_loss:.4f}\tacc.={train_acc*100:.2f}%\t'
                   f'IoU={train_iou:.3f}\t|\tval loss={mean_val_loss:.4f}\tval accuracy={mean_val_acc*100:.2f}%\t'
                   f'val IoU={mean_val_iou:.3f}\tval f1={mean_val_f1:.3f}\t'
                   f'per class acc={list(np.around(mean_val_class_acc*100, 2))}\t|\t'
                   f'time per epoch: {int(time_elapsed//60):02d} min {int(time_elapsed%60):02d} sek\n')
        logger.info(message)
        print(message)

        # save model
        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            FILE = f'pretrained models/' \
                   f'model_{epoch+1}_epochs-{len(train_dataloader)}_batches-batchsize_{(len(train_dataloader)/1000):.0f}k.pth'
            torch.save(model.state_dict(), FILE)
            logger.info(f'New best validation accuracy.\nSaved Model to {FILE}')

    # save last calculated model
    FILE = f'pretrained models/model_{epoch+1}_epochs-{len(train_dataloader)}_batches-batchsize_{(len(train_dataloader)/1000):.0f}k.pth'
    torch.save(model.state_dict(), FILE)
    logger.info(f'New best validation accuracy.\nSaved Model to {FILE}')
