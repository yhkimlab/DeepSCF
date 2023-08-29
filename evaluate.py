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

def evaluate(args, device, test_list, model, train_list = None):

    def load_hdf5(path):
        f = h5py.File(path,'r')
        target = f['target'][:]
        feature = f['feature'][:]
        f.close()
        x = torch.from_numpy(feature).float()
        y = torch.from_numpy(target).float()

        return x.unsqueeze(0), y.unsqueeze(0)

    model.eval()


    # logging
    Logger = log.logger()

    # evaluate test loss
    
    # loss function
    loss = nn.L1Loss(reduction='sum')
    ne = 0 # total number of electron
    mae = 0 # mean absolute error

    with torch.no_grad():
        for path in test_list:

            print(path)
            data, target = load_hdf5(path)
            data, target = data.to(device), target.to(device)
            output = model(data)
            mae += loss(output, target).item()
            ne += torch.sum(target).item()
            p_error = 100 * loss(output, target).item() / torch.sum(target).item()

            # logging
            Logger.update(path=path, loss=p_error)

    # save log
    Logger.save()

    percentage_error = mae/ne*100
    f = open('summary.txt','w')
    f.write(f'Test percentage error: {percentage_error} \n')
    print(f'Percentage error: {percentage_error} \n')          

    if train_list != None:
        ne = 0 # total number of electron
        mae = 0 # mean absolute error

        with torch.no_grad():
            for path in train_list:
 
                print(path)
                data, target = load_hdf5(path)
                data, target = data.to(device), target.to(device)
                output = model(data)
                mae += loss(output, target).item()
                ne += torch.sum(target).item()

        percentage_error = mae/ne*100
        f.write(f'Train percentage error: {percentage_error} \n')
        print(f'Percentage error: {percentage_error} \n')



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
        evaluate(args, device, target_list, model)
 
    elif path_data.split('.')[-1] == 'h5':
        target_list = [path_data]
        evaluate(args, device, target_list, model)

    else:
        with open(path_data,'rb') as f:
            data_list = pickle.load(f)
        target_list = data_list['test']
        train_list = data_list['train']
        evaluate(args, device, target_list, model, train_list)

if __name__=='__main__':
    main()
