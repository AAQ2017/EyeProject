import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from torch import nn
from torch.nn import functional as F

def train_model(model, batch_size, scheduler, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image']
                masks = sample['mask']
                masks[masks == 255] = 1
                masks[masks == 192] = 2
                masks[masks == 128] = 3
                masks[masks == 85] = 4
                masks[masks == 64] = 5

                image_copy = np.zeros((batch_size, 8, 256, 256)).astype(np.float32)
                image_copy[:, 1:, :, :] = inputs[:, 0:8 - 1, :, :]
                image_res = inputs - image_copy
                image_res[:, 0, :, :] = 0
                image_res = np.abs(image_res)

                # label -> res
                label_copy = np.zeros((batch_size, 8, 256, 256)).astype(np.float32)
                label_copy[:, 1:, :, :] = masks[:, 0:8 - 1, :, :]

                label_res = masks - label_copy
                label_res = np.abs(label_res)

                # label_res[np.where(label_res == 0)] = 0
                # label_res[np.where(label_res != 0)] = 1

                inputs = inputs.unsqueeze(1)
                image_res = image_res.unsqueeze(1)
                inputs = torch.cat((inputs, image_res), dim=1)

                inputs = inputs.to(device)
                masks = masks.to(device)
                label_res = label_res.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    seg, res = model(inputs)

                    loss = criterion(seg, masks.long()) + criterion(res, label_res.long())

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

            scheduler.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(
                phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.load_state_dict(best_model_wts), os.path.join(bpath, 'MRI_latest.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
