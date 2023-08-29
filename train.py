import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import log, data
from model import ResUNet
import hydra
import os
from omegaconf import DictConfig, OmegaConf


def save_data(original_path, working_path):
    os.system(f'cp {original_path}/*.py {working_path}')
    os.system(f'cp -r {original_path}/utils {working_path}')

def train(args, model, device, train_loader, optimizer, loss_function, epoch):

    model.train()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % args.logger.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))

    return train_loss


def test(model, device, test_loader, loss_function):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #target = target.squeeze(dim=1)
            test_loss += loss_function(output, target).item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss


def reference(device, test_loader, loss_function):

    reference_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.squeeze(dim=1)
            reference_loss += loss_function(output, target).item() # sum up batch loss

    reference_loss /= len(test_loader.dataset)
    print('\nReference loss: {:.4f}\n'.format(reference_loss))

    return reference_loss


@hydra.main(config_path="config", config_name="train")
def main(args: DictConfig):

    # save current codes to working directory
    save_data(original_path = hydra.utils.get_original_cwd(),
              working_path = os.getcwd()
             )

    # gpu/cpu
    device = torch.device(f'cuda:{args.gpu:d}' if torch.cuda.is_available() else 'cpu') 

    # model & optimizer
    model = ResUNet(args.model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.optimizer.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.optimizer.step_size,
                                          gamma= args.optimizer.gamma)

    if (args.load.reoptimization):
        # load saved model & optimizer 
        path = hydra.utils.get_original_cwd()+'/'+args.load.model
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # dataloader
        train_loader, test_loader = data.load_dataloader(args)

        # epoch index
        start_epoch = int(checkpoint['epoch'])+1

    elif (args.load.transfer):
        # load saved model & optimizer
        path = hydra.utils.get_original_cwd()+'/'+args.load.model
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

        # dataloader
        dummy_loader, test_loader = data.load_dataloader(args)
        train_loader, dummy_loader = data.create_dataloader(args)

        # epoch index
        start_epoch = 1

    else:
        # dataloader
        train_loader, test_loader = data.create_dataloader(args)

        # epoch index
        start_epoch = 1

    # loss function
    loss = nn.MSELoss(reduction='sum')

    # logging
    Logger = log.logger(path=args.logger.path)

    # summary of model
    num_parameters = sum(p.numel() for p in model.parameters())
    Logger.summary_model(args.model, num_parameters) 

    # optimize
    for epoch in range(start_epoch, args.train.epochs+1):
        train_loss = train(args, model, device, train_loader, optimizer, loss, epoch)
        test_loss = test(model, device, test_loader, loss)
        Logger.update(epoch=epoch, train_loss=train_loss, test_loss=test_loss)
        scheduler.step()

        # check point
        if epoch % args.save.interval == 0:
            torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                      },f'{epoch}_model.pt')
            Logger.save()

    # save results
    if (args.save.model):
        torch.save(model.state_dict(),"model.pt")

if __name__=='__main__':
    main()
