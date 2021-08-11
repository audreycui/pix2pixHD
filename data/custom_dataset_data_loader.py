from torch.utils.data import Dataset, DataLoader
from data.base_data_loader import BaseDataLoader
import torch 

import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pbw_utils import zdataset, show, labwidget, paintwidget, renormalize, nethook, imgviz, pbar
from pbw_utils.stylegan2 import load_seq_stylegan

from PIL import Image

def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

class StyleGANDatasetDataLoader(BaseDataLoader): 
    def name(self): 
        return 'StyleGANDatasetDataLoader'
    
    def initialize(self, opt): 
        BaseDataLoader.initialize(self, opt)
        self.dataset = StyleGANDataset()
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            #num_workers=int(opt.nThreads))
            num_workers=0)
        
    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
        
class StyleGANDataset(Dataset): 
    def __init__(self, 
                 dset='bedroom', 
                 batch_size = 1
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        self.batch_size = batch_size
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        #self.segmodel, self.seglabels = load_segmenter()
        self.color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()[:,None,None]
        self.frac = ((float(100) * 2 - 100) / 100.0)
        self.num = 0

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        z = torch.randn(self.batch_size, 512, device='cuda')
        original = self.model(z)[0]
        adjusted = self.get_lit_scene(z, self.frac, self.light_layer, self.light_unit)[0]
        data = {'label': original, 'image': adjusted, 'inst': 0, 'feat': 0, 'path': f'bedroom_{self.num}'}
        self.num += 1
        return data
    
    def get_lit_scene(self, z, amount, layername, unitnum):
        def change_light(output):
            output.style[:, int(unitnum)] = 10 * amount
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
