from dataset import dataset, dataloader
from model import Mydense121
from parser_my import args
from utils.plot import plot_roc, plot_confusion_matrix, plot_tensor
import torch
import numpy as np
from torchmetrics.functional import auroc, roc, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
device = args.device

classes = dataset['test'].classes
num_class = len(classes)
model = Mydense121(num_class=num_class, weights=True)
model.load_state_dict(torch.load("checkpoint/best_model_0.9976acc_32epochs.pth"))
model = model.to(device)


def test_model():
    total_preds = torch.empty(0)
    total_targets = torch.empty(0)
    error_data = torch.empty(0)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader['test']:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total_preds = torch.concat((total_preds, outputs.cpu()))
            total_targets = torch.concat((total_targets, targets.cpu()))

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            index = torch.nonzero(predicted != targets)
            error_data = torch.concat((error_data, inputs[index].cpu()))

    error_data = error_data.squeeze(1)
    plot_tensor(error_data)
    print(error_data.shape)
    print('Correct Prediction: {:d}  Total Images: {:d}'.format(correct, total))
    print('Test Accuracy = {:f}'.format(correct / total))

    return total_preds, total_targets

preds, targets = test_model()


auc = auroc(preds, targets.int(), task='multiclass', num_classes=num_class, average='none')
auc_macro = auroc(preds, targets.int(), task='multiclass', num_classes=num_class, average='macro')
fpr, tpr, thresholds = roc(preds, targets.int(), task='multiclass', num_classes=num_class)
f1 = f1_score(preds, targets.int(), task='multiclass', num_classes=num_class, average='macro')
conf_matrix = confusion_matrix(preds, targets.int(), task='multiclass', num_classes=num_class)


all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_class

plot_roc(fpr, tpr, all_fpr, mean_tpr, auc, auc_macro, num_class)
plot_confusion_matrix(conf_matrix, num_class, classes)


print(auc) # tensor([1.0000, 1.0000, 0.9999])
print(auc_macro) # tensor(0.9999)
print(f1) # tensor(0.9945)

# Correct Prediction: 403  Total Images: 405
# Test Accuracy = 0.995062
