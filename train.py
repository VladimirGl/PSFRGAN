from utils.timer import Timer
from utils.logger import Logger
from utils import utils

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from data.celebahqmask_dataset import CelebAHQMaskDataset
from models.networks import MParseNet

import yaml
import torch 
import os
import torch.multiprocessing as mp
from albumentations.core.serialization import from_dict
from torch.utils.data import DataLoader

def train(opt):
    with open(opt.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    train_aug = from_dict(hparams["train_aug"])
    degrd_aug = from_dict(hparams["degradation_aug"])
    hr_aug = from_dict(hparams["hr_aug"])

    dataset = CelebAHQMaskDataset(opt, train_aug, degrd_aug, hr_aug)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True)
    dataloader_size = len(dataloader)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)

    mparse_model = MParseNet(hparams["model"]).cuda()
    model.netP = mparse_model
    
    logger = Logger(opt)
    timer = Timer()

    single_epoch_iters = (dataset_size // opt.batch_size)
    total_iters = opt.total_epochs * single_epoch_iters 
    cur_iters = opt.resume_iter + opt.resume_epoch * single_epoch_iters
    start_iter = opt.resume_iter
    print('Start training from epoch: {:05d}; iter: {:07d}'.format(opt.resume_epoch, opt.resume_iter))
    for epoch in range(opt.resume_epoch, opt.total_epochs + 1):    
        for i, data in enumerate(dataloader, start=start_iter):
            cur_iters += 1
            logger.set_current_iter(cur_iters)
            # =================== load data ===============
            model.set_input(data, cur_iters)
            timer.update_time('DataTime')
    
            # =================== model train ===============
            model.forward(), timer.update_time('Forward')
            model.optimize_parameters() 
            loss = model.get_current_losses()
            loss.update(model.get_lr())
            logger.record_losses(loss)
            timer.update_time('Backward')

            # =================== save model and visualize ===============
            if cur_iters % opt.print_freq == 0:
                print('Model log directory: {}'.format(opt.expr_dir))
                epoch_progress = '{:03d}|{:05d}/{:05d}'.format(epoch, i, single_epoch_iters)
                logger.printIterSummary(epoch_progress, cur_iters, total_iters, timer)
    
            if cur_iters % opt.visual_freq == 0:
                visual_imgs = model.get_current_visuals()
                logger.record_images(visual_imgs)
            
            if cur_iters % opt.save_iter_freq == 0:
                print('saving current model (epoch %d, iters %d)' % (epoch, cur_iters))
                save_suffix = 'iter_%d' % cur_iters 
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks(save_suffix, info)

            if cur_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, iters %d)' % (epoch, cur_iters))
                info = {'resume_epoch': epoch, 'resume_iter': i+1}
                model.save_networks('latest', info)

            if i >= single_epoch_iters - 1:
                start_iter = 0
                break

            #  model.update_learning_rate()
            if opt.debug: break
        if opt.debug and epoch >= 0: break 
    logger.close()

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train(opt)
    
