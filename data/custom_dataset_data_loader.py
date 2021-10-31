from torch.utils.data import Dataset, DataLoader
from data.base_data_loader import BaseDataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torch 
import torch.nn.functional as F

import os, sys, inspect
from os import path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pbw_utils import zdataset, show, labwidget, paintwidget, renormalize, nethook, imgviz, pbar
from pbw_utils.stylegan2 import load_seq_stylegan

from PIL import Image
import random
import numpy as np


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
        if opt.alternate_train: 
            on_dataset = AlternateStyleGANDataset()
            off_dataset = AlternateStyleGANDataset(reverse = True)
            self.dataset = torch.utils.data.ConcatDataset([on_dataset, off_dataset])
 
        else: 
            if opt.n_stylechannels > 1:
                self.dataset = MultichannelStyleGANDataset(opt.n_stylechannels)
            else:
                if opt.isTrain: 
                    loc_map = opt.use_location_map
                    bootstrap = opt.lamp_off_bootstrap
                    self.dataset = StyleGANDataset(loc_map=loc_map, bootstrap=bootstrap)
                else: 
                    self.dataset = StyleGANDataset(frac_one = opt.frac_one)
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
                 loc_map=False, 
                 bootstrap=False,
                 dset='bedroom', 
                 debug = False, 
                 frac_one = False, 
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        #self.segmodel, self.seglabels = load_segmenter()
        self.color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()[:,None,None]
        self.num = 0
        self.use_loc = loc_map
        self.bootstrap = bootstrap
        self.frac_one = frac_one
        if self.use_loc: 
            self.loc_frac = [2]
            self.kernel_dim = 7
            self.blur_kernel = (1/2**(self.kernel_dim))*torch.ones(self.kernel_dim, self.kernel_dim)
            self.blur_kernel = self.blur_kernel.repeat(1, 3, 1, 1).cuda()

        self.debug = debug
        if self.debug: 
            if path.exists('fixed_z.pt'): 
                self.fixed_z = torch.load('fixed_z.pt')
            else: 
                self.fixed_z = torch.randn(self.batch_size, 512, device='cuda')
                torch.save(self.fixed_z, 'fixed_z.pt') 
                #save fixed z 

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
        
        if self.debug: 
            z = self.fixed_z
            
        original = self.model(z)[0]
        input_img = original
        
        if self.bootstrap: 
            frac = np.random.rand(1)
            adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
            input_img = adjusted
            output_img = original
            frac = -frac
        elif self.debug: 
            output_img = original
            frac = 0
        else: 
            frac = np.random.rand(1)*2-1
            if self.frac_one: 
                frac = [1]
            adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
            output_img = adjusted
        

        feat = 0
        if self.use_loc: 
            diff = original.unsqueeze(0) - self.get_lit_scene(z, self.loc_frac, self.light_layer, self.light_unit)
            blur = F.conv2d(diff, self.blur_kernel)
            feat = ((blur) > 0.7).float() * 1
            
            
        data = {'label': input_img, 'image': output_img, 'inst': 0, 'feat': feat, 'path': f'bedroom_{self.num}', 'frac': frac}       
        
        self.num += 1
        return data
    
    def get_lit_scene(self, z, frac, layername, unitnum):
        def change_light(output):
            output.style[:, int(unitnum)] = 10 * frac[0]
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
        
class MultichannelStyleGANDataset(Dataset): 
    def __init__(self,
                 num_stylechannels, 
                 dset='bedroom', 
                 debug = False,  
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.layers = ['layer8', 'layer8']
        self.units = [265, 397] #[lamp, window]

        self.color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()[:,None,None]
        self.num = 0
        self.num_stylechannels = num_stylechannels
        
        self.debug = debug
        if self.debug: 
            if path.exists('fixed_z.pt'): 
                self.fixed_z = torch.load('fixed_z.pt')
            else: 
                self.fixed_z = torch.randn(self.batch_size, 512, device='cuda')
                torch.save(self.fixed_z, 'fixed_z.pt') 
                #save fixed z 

    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
        
        if self.debug: 
            z = self.fixed_z
            
        original = self.model(z)[0]
        frac = np.random.rand(self.num_stylechannels)*2-1
        adjusted = self.get_lit_scene(z, frac, self.layers, self.units)[0]
        
        if self.debug: 
            adjusted = original
            frac = 0
            
        data = {'label': original, 'image': adjusted, 'inst': 0, 'feat': 0, 'path': f'bedroom_{self.num}', 'frac': frac}
        self.num += 1
        return data
    
    def get_lit_scene(self, z, fracs, layers, units):
        #TODO: modify for multiple layers
        layername = layers[0]
        def change_light(output):
            for frac, unit in zip(fracs, units): 
                output.style[:, int(unit)] = 10 * frac
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)
        
        
class AlternateStyleGANDataset(Dataset): 
    def __init__(self, 
                 dset='bedroom', 
                 reverse = False
                ): 
        self.model = load_seq_stylegan('bedroom', mconv='seq', truncation=0.90)
        nethook.set_requires_grad(False, self.model)
        self.light_layer = 'layer8'
        self.light_unit = 265
        self.reverse = reverse
        self.num=0
    def __len__(self):
        return 5000

    def __getitem__(self, index):
        
        z = torch.randn(1, 512, device='cuda')
            
        original = self.model(z)[0]
        input_img = original
        
        frac = np.random.rand(1)
        adjusted = self.get_lit_scene(z, frac, self.light_layer, self.light_unit)[0]
        output_img = adjusted
        
        if self.reverse: 
            frac = -frac
            input_img = adjusted
            output_img = original
 
        data = {'label': input_img, 'image': output_img, 'inst': 0, 'feat': 0, 'path': f'bedroom_{self.num}', 'frac': frac}       
        self.num+=1
        return data
    
    def get_lit_scene(self, z, frac, layername, unitnum):
        def change_light(output):
            output.style[:, int(unitnum)] = 10 * frac[0]
            return output
        with nethook.Trace(self.model, f'{layername}.sconv.mconv.modulation', edit_output=change_light):
            return self.model(z)

