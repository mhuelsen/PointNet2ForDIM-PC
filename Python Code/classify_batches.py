from tools.dataset import *
from tools.pointnet_utils import *
from tools.validation import *
import torchmetrics
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

"""
This python code is part of a master thesis with the title: 
Analysis of deep learning methods for semantic segmentation of photogrammetric point clouds from aerial images

© Markus Hülsen, Matr.-Nr. 6026370
Date: 22.08.2023

This script classifies the batches in a folder and concatenate them to a single point cloud
"""


if __name__ == '__main__':

    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # hyperparameters
    num_classes = 5
    batch_size = 6

    # load model
    model = PointNet2_v2(num_classes).to(device)
    model.load_state_dict(torch.load('pretrained models/model_39_epochs-all_batches-fl_g4_AdamW_lr-scheduler_6-atts.pth'))

    # load pointcloud
    path = r'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/validation/'
    val_dataset = PointcloudDataset_v2(path, augmentation=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # initialize DataFrames
    df_pc = pd.DataFrame()
    df_probs = pd.DataFrame()

    counter = 0

    # iterate throw batches
    with (torch.no_grad()):
        for i, (inputs, labels) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            # change labels in to a numeric ordered representation
            labels = labels.to(device)
            inputs = inputs.to(device)
            labels[labels == 2] = 0
            labels[labels == 6] = 1
            labels[labels == 20] = 2
            labels[labels == 25] = 3
            labels[labels == 30] = 4

            # forward pass
            outputs = model(inputs).to(device)

            # get predicted labels
            _, predictions = torch.max(torch.reshape(outputs, (-1, 5)), 1)
            probs = F.softmax(outputs, dim=2).to(torch.device('cpu'))

            # iterate throw batches
            for j in range(probs.size(0)):
                # get current point cloud and update it to the point cloud dataframe
                df_pc = pd.concat([df_pc, val_dataset.lst_pointclouds[counter].df_pc.set_index('point_index')])
                df_pc = df_pc.groupby(df_pc.index).max()

                # get the probabilities for every class and save them with respect to the point index
                df_temp = pd.DataFrame(data=probs[j, :, :].to(torch.device('cpu')),
                                       index=val_dataset.lst_pointclouds[counter].df_pc['point_index'],
                                       columns=[2, 6, 20, 25, 30])

                # add a counter to count the number of votes
                df_temp.loc[:, 'count'] = 1
                # add the probabilities to the global DataFrame and sum the results up
                df_probs = pd.concat([df_probs, df_temp])
                df_probs = df_probs.groupby(df_probs.index).sum()

                counter += 1

        # calculate the mean probability for each class
        df_probs.loc[:, [2, 6, 20, 25, 30]] = df_probs.divide(df_probs.loc[:, 'count'], axis=0)
        # get maximum of the probability to define the predicted class
        df_pc['pred_class'] = df_probs.loc[:, [2, 6, 20, 25, 30]].idxmax(axis=1)
        # get probability of the predicted class
        df_pc['probability'] = df_probs.loc[:, [2, 6, 20, 25, 30]].max(axis=1)
        # add the number of votes for the selected class
        df_pc['count_votes'] = df_probs['count']

        # export the results
        ex_path = f'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/results/validation.laz'

        df_pc.insert(3, 'intensity', 0)
        save_df_to_las(df_pc, ex_path)

        # calculate confusion matrix if ground truth is available
        if 'classification' in df_pc.columns:
            # generate Confusion matrix
            drop_idx = df_pc.loc[df_pc['classification'] == 0].index
            df_pc = df_pc.drop(drop_idx)
            df_probs = df_probs.drop(drop_idx)

            conf_mat = confusion_matrix(df_pc.classification, df_pc.pred_class, labels=[2, 6, 20, 25, 30], normalize=None)
            class_names = ['Ground', 'Building', 'Vegetation', 'humanmade', 'bridge']

            # export confusion matrix
            plot_confusion_matrix(conf_mat, class_names, ex_path.replace('laz', 'png'), show=False)

            # clear plot
            plt.clf()

            # make ROC Curves
            labels = df_pc.classification.to_numpy().copy()
            labels[labels == 2] = 0
            labels[labels == 6] = 1
            labels[labels == 20] = 2
            labels[labels == 25] = 3
            labels[labels == 30] = 4

            labels = torch.Tensor(labels).to(device)
            ROC = torchmetrics.ROC(task='multiclass', num_classes=num_classes, thresholds=9).to(device)
            fpr, tpr, thresh = ROC(torch.Tensor(df_probs.loc[:, [2, 6, 20, 25, 30]].values).to(device),
                                   torch.Tensor(labels.to(torch.int64)))
            ROC_path = r'../Daten/Datensatz_H3D/DIM_2022/8 - PointNet++/results/ROC_' + \
                       ex_path.split('/')[-1].replace('laz', 'png')
            # export ROC-Plot
            ROC_Plot(fpr, tpr, num_classes, class_names, ROC_path, show=False)

            # calculate accuracy
            acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
            labels = torch.Tensor(df_pc.classification.values).to(device)
            pred_classes = torch.Tensor(df_pc.pred_class.values).to(device)
            print(f'Accuracy: {acc(pred_classes, labels)*100:.2f}%')
