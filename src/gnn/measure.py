import torch
import torch.nn as nn
import tqdm
import numpy as np

from sklearn import metrics
from torch_geometric.data import DataLoader


from model import GIN0WithJK, GIN0, DGCNN
from dataset import CFGDataset_Normalized_After_BERT, CFGDataset_MAGIC


def split_pred_label(predictions, labels):
    target_class = 0

    sliced_predictions = []
    sliced_labels = []
    slice_index = 0

    for idx, label in enumerate(labels):
        if label != target_class:
            sliced_labels.append(labels[slice_index:idx])
            sliced_predictions.append(predictions[slice_index:idx])
            slice_index = idx
            target_class += 1
        
        if idx == len(labels) - 1:
            sliced_labels.append(labels[slice_index:])
            sliced_predictions.append(predictions[slice_index:])
    
    return sliced_predictions, sliced_labels


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # big2015_dataset = CFGDataset_Normalized_After_BERT(
    #     root='/home/wubolun/data/malware/big2015/further',
    #     vocab_path='/home/wubolun/data/malware/big2015/further/set_0.5_pair_30/normal.vocab',
    #     seq_len=64)

    big2015_dataset = CFGDataset_MAGIC(root='/home/wubolun/data/malware/big2015/further')

    recalls = []
    precisions = []
    f1s = []

    for k in range(5):

        train_idx, val_idx = big2015_dataset.train_val_split(k)
        val_dataset = big2015_dataset[val_idx]
        val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False, num_workers=5)

        # model = DGCNN(num_features=20, num_classes=big2015_dataset.num_classes)
        model = GIN0WithJK(num_features=20, num_layers=5, hidden=128, num_classes=big2015_dataset.num_classes)
        model.load_state_dict(
            torch.load('result/magic-gin0jk-5-128/gin0jk_{}.pth'.format(k), map_location='cuda:0'))
        model = model.to(device)
        model.eval()

        # criterion = nn.NLLLoss()

        predictions = []
        labels = []

        running_loss = 0.0
        for data in tqdm.tqdm(val_loader):
            data = data.to(device)
            out = model(data)

            predictions.extend(out.argmax(dim=1).tolist())
            labels.extend(data.y.tolist())
            # loss = criterion(out, data.y)
            # running_loss += loss.item() * data.y.size(0)

        # print('k: {}, loss: {}'.format(k, running_loss/len(val_loader.dataset)))

        recall = list(metrics.recall_score(labels, predictions, average=None))
        precision = list(metrics.precision_score(labels, predictions, average=None))
        f1 = list(metrics.f1_score(labels, predictions, average=None))

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    # average on 5-fold
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f1s = np.array(f1s)

    mean_recall = list(np.mean(recalls, 0))
    mean_precision = list(np.mean(precisions, 0))
    mean_f1 = list(np.mean(f1s, 0))

    print('recall: {}'.format(mean_recall))
    print('precision: {}'.format(mean_precision))
    print('f1 score: {}'.format(mean_f1))

    print('average recall: {}'.format(np.mean(np.array(mean_recall))))
    print('average precision: {}'.format(np.mean(np.array(mean_precision))))
    print('average f1 score: {}'.format(np.mean(np.array(mean_f1))))