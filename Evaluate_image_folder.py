from Model import SiameseConvNet, distance_metric
from torch import load
import torch
import numpy as np
from Dataloaders import TestDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

model = SiameseConvNet()
model.load_state_dict(load(open('Models/model_large_epoch_20', 'rb'), map_location=device))


def compute_accuracy_roc(predictions, labels):
    dmax = np.max(predictions)
    dmin = np.min(predictions)
    actual_positive = np.sum(labels == 1)
    actual_negative = np.sum(labels == 0)
    step = 0.001
    max_acc = 0

    d_optimal = 0
    for d in np.arange(dmin, dmax + step, step):
        idx1 = predictions.ravel() <= d
        idx2 = predictions.ravel() > d

        tp = np.sum(labels[idx1] == 1)
        tn = np.sum(labels[idx2] == 0)
        fp = np.sum(labels[idx2] == 1)
        # fn = np.sum(labels[idx1] == 0)

        tpr = float(tp) / actual_positive  # recall
        tnr = float(tn) / actual_negative
        acc = 0.5 * (tpr + tnr)

        if acc > max_acc:
            max_acc = acc
            d_optimal = d
            precision = float(tp)/(tp+fp)
            recall = tpr

    return max_acc, d_optimal, precision, recall

def test():
    model.eval()
    global batch_avg_acc, batch_avg_d, n_batch, batch_avg_precision, batch_avg_recall

    test_dataset = TestDataset()
    loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    for batch_index, data in enumerate(loader):
        A = data[0]
        B = data[1]
        labels = data[2].long()

        f_a, f_b = model.forward(A, B)
        dist = distance_metric(f_a, f_b)

        acc, d, precision, recall = compute_accuracy_roc(dist.detach().numpy(), labels.detach().numpy())
        print('Max accuracy for batch {} = {} at d = {}'.format(batch_index, acc, d))
        batch_avg_acc += acc
        batch_avg_d += d
        n_batch += 1
        batch_avg_precision += precision
        batch_avg_recall += recall

batch_avg_precision = 0
batch_avg_recall = 0
batch_avg_acc = 0
batch_avg_d = 0
n_batch = 0

print('Validating:')
test()
print('Avg acc across all batches={} at d={}'.format(batch_avg_acc / n_batch, batch_avg_d / n_batch))
print('Avg precision across all batches={} and avg recall={}'.format(batch_avg_precision / n_batch, batch_avg_recall / n_batch))