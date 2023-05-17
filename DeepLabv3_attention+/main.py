import torch.optim
from sklearn.metrics import roc_auc_score, f1_score
from model import createDeepLabv3
from trainer import train_model
import datahandler
import os
import torch

bpath = "CFExp"
data_dir = "dataset"
epochs = 25
batchsize = 4

def perform_main():
    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(data_dir, batch_size=batchsize)
    model = createDeepLabv3(outputchannels=6)
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    # Create the experiment directory if not present
    if not os.path.isdir(bpath):
        os.mkdir(bpath)

    # Specify the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Specify the evalutation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}
    trained_model = train_model(model, criterion, scheduler, dataloaders,
                                optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)

    # Save the trained model
    torch.save(model, os.path.join(bpath, 'model.pt'))

if __name__ == "__main__":
    perform_main()