from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

cudnn.benchmark = True

# modify the following code to 
datadir = '/media/kaixuan/DATA/Papers/Code/Data/Reflection/'

datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')
datadir_unaligned = join(datadir, 'unaligned', 'unaligned_train250')

train_dataset = datasets.CEILDataset(datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size)
train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)

train_dataset_unaligned = datasets.CEILTestDataset(datadir_unaligned, enable_transforms=True, flag={'unaligned':True}, size=None)

train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_unaligned, train_dataset_real], [0.25,0.5,0.25])


train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches, 
    num_workers=opt.nThreads, pin_memory=True)


engine = Engine(opt)
"""Main Loop"""
def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        util.set_opt_param(optimizer, 'lr', lr)


set_learning_rate(1e-4)
while engine.epoch < 80:
    if engine.epoch == 65:
        set_learning_rate(5e-5)
    if engine.epoch == 70:
        set_learning_rate(1e-5)
        
    engine.train(train_dataloader_fusion)
