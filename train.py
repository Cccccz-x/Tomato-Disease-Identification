from dataset import dataset, dataloader
from model import Mydense121
from parser_my import args
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import os
from glob import glob
from tqdm import tqdm
from utils.EarlyStopping import EarlyStopping
import warnings
warnings.simplefilter('ignore')


device = args.device
print(device)

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
print(dataset_sizes)
c2id = dataset['train'].class_to_idx
print(c2id)


def train_model(model, criterion, optimizer, name, num_epochs=40):
    writer = SummaryWriter('log')

    #Creating a folder to save the model performance.
    try:
        os.mkdir(f'./{name}')
    except:
        print('existed')

    since = time.time()

    best_acc = 0.0

    ES = EarlyStopping()
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Train
        model.train()  # Set model to training mode
        train_loss = 0
        num_corrects = 0
        
        i = int(len(dataset['train']) / args.batch_size)
            
        for _ in tqdm(range(i)):
            inputs, labels = next(iter(dataloader['train']))

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            num_corrects += torch.sum(preds == labels.data)
            
        train_loss = train_loss / dataset_sizes['train']
        train_acc = num_corrects.double() / dataset_sizes['train']

        print('train loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))


        # Val
        
        model.eval()  # Set model to training mode
        val_loss = 0
        num_corrects = 0
        with torch.no_grad():
            i = int(len(dataset['val']) / args.batch_size)
            for _ in tqdm(range(i)):
                inputs, labels = next(iter(dataloader['val']))

                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                num_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / dataset_sizes['val']
        ES(val_loss)
        if ES.early_stop:
            break
        val_acc = num_corrects.double() / dataset_sizes['val']

        print('val loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), './{}/best_model_{:.4f}acc_{}epochs.pth'.format(name, val_acc, epoch))
            

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


model = Mydense121(num_class=len(c2id), weights=True)
name = model._get_name()
go = 1 #继续上次的训练
if go:
	model.load_state_dict(torch.load(f'./{name}/'+sorted(os.listdir(f'./{name}'))[-1]))
print(model)

criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

train_model(model, criterion, optimizer, name, num_epochs=args.epochs)
