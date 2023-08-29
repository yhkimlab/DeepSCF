import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hydra
import os
from omegaconf import DictConfig, OmegaConf
from model import ResUNet
from utils import data, log
from utils.unit import ang2bohr
import siestaio as io
import pickle
import h5py
import numpy as np
import glob

def convert(args, device, test_list, model):

    def load_hdf5(path):
        f = h5py.File(path,'r')
        target = f['target'][:]
        feature = f['feature'][:]
        f.close()
        x = torch.from_numpy(feature).float()
        y = torch.from_numpy(target).float()

        return x.unsqueeze(0), y.unsqueeze(0)

    model.eval()

    # loss function
    loss = nn.MSELoss(reduction='sum')

    # get reference mesh data
    cell0 = np.array(args.target.cell)
    mesh0 = np.array(args.target.mesh)

    # logging
    Logger = log.logger()

    # the most accurate/inaccurate model
    max_loss = 0
    min_loss = 100

    # convert predicted mesh data
    with torch.no_grad():
        for path in test_list:
            data, target = load_hdf5(path)
            data, target = data.to(device), target.to(device)
            output = model(data)
            initial = args.model.std * data[:,1,:,:,:] + args.model.mean

            # keep good/bad model
            loss_val = loss(output, target).item()
            if (loss_val > max_loss):
                max_loss = loss_val
                good_model = path
            elif (loss_val < min_loss):
                min_loss = loss_val
                bad_model = path
            
            # logging
            Logger.update(path=path, loss=loss(output, target).item())

            print(f'Path: {path}')
            print(f'Initial loss: {loss(initial, target).item()}')
            print(f'ML loss: {loss(output, target).item()}')

            # Tensor object to Numpy array
            output = output.cpu().numpy()
            name = path.split('/')[-1].split('.h5')[0] + '.RHO'

            # write density in SIESTA format
            mesh = np.array(np.shape(output)[1:])
            cell = np.zeros((3,3))
            cell[0] = cell0[0] * mesh[0] / mesh0[0]
            cell[1] = cell0[1] * mesh[1] / mesh0[1]
            cell[2] = cell0[2] * mesh[2] / mesh0[2]
            io.writeGrid(name, cell, mesh, output)

    # save log
    Logger.save()

    # 


@hydra.main(config_path="config", config_name="predict")
def main(args: OmegaConf):

    # gpu/cpu
    device = torch.device(f'cuda:{args.gpu:d}' if torch.cuda.is_available() else 'cpu')

    # model
    model = ResUNet(args.model).to(device)

    # load model & datalist
    path_model = hydra.utils.get_original_cwd() + '/' + args.load.model
    checkpoint = torch.load(path_model)

    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # target path
    path_data = hydra.utils.get_original_cwd() + '/' + args.target.path

    if os.path.isdir(path_data):
        target_list = glob.glob(path_data + '/*.h5')
    elif path_data.split('.')[-1] == 'h5':
        target_list = [path_data]
    else:
        with open(path_data,'rb') as f:
            data_list = pickle.load(f)
        target_list = data_list['test']

    # convert to SIESTA RHO data
    convert(args, device, target_list, model)



if __name__=='__main__':
    main()
