import os
from omegaconf import OmegaConf

class logger():

    def __init__(self, path = './log.txt', mode = 'w'):

        self.path = path
        if not os.path.isfile(path):
            f = open(path, 'w')
            f.close()
        else:
            f = open(path, 'w')
            f.close()
        self.log = ""

    def summary_model(self, args, num_parameter):

        f = open('./summary.txt','w')

        dicts = OmegaConf.to_object(args)
        for k, i in dicts.items():
            f.write(f'{k}: {i}\n')
        f.write(f'Number of parameters {num_parameter}\n')
        f.close()


    def update(self, **kwags):

        if 'path' in kwags:
            path = kwags['path']
            loss =  kwags['loss']
            txt = f"Path: {path} loss = {loss} \n"

        elif 'epoch' in kwags:
            epoch = kwags['epoch']
            train_loss =  kwags['train_loss']
            test_loss = kwags['test_loss']
            txt = f"Epoch {epoch}: train loss = {train_loss} test_loss = {test_loss} \n"

        self.log += txt

    def save(self):

        f = open(f'{self.path}', 'a')
        txt = self.log
        f.write(txt)
        f.close()

        # refresh the log
        self.log = ""
