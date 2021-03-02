import argparse
import collections

import numpy as np
import pandas as pd

import torch, os
from utils import makedir
import torch.optim as optim
from tqdm import tqdm
from dataset import VinBigDataset
from torchvision import transforms
from utils import split_data, class_id_name_mapping

from retinanet import model
from dataset import collate
# from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
#     Normalizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from retinanet import coco_eval
# from retinanet import csv_eval


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def train(epoch_num, train_loss_per_epoch):
        loss_per_epoch_train = []
        for iter_num, data in tqdm(enumerate(dataloader_train)):
                #print(data['img'].size(),' ',data['annot'].size())
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()


                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss.item()))

                train_loss_per_epoch+=loss.item()

                loss_per_epoch_train.append(float(loss))

                if iter_num%print_every==0:
                    print('Epoch:{} | Batch: {} | cls_loss: {:.4f} | reg_loss:{:.4f} | running loss:{:.4f}'.format(epoch_num, iter_num, float(classification_loss.item()), float(regression_loss.item()), train_loss_per_epoch/(iter_num+1)))

                del classification_loss
                del regression_loss


        avg_loss = train_loss_per_epoch/len(dataloader_train)
        train_epoch_loss.append(avg_loss)
        print('For epoch:{} total train loss:{:.4f}'.format(epoch_num, avg_loss))
        torch.save({'epoch':epoch_num,'model_state_dict':retinanet.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),'scheduler_dict':scheduler.state_dict(),'train_loss':avg_loss}, os.path.join(save_path_train, 'epoch_{}_{:.4f}_state_dict.pt'.format(epoch_num, avg_loss)))
        writer.add_scalar('training_loss', avg_loss, epoch_num+1)
        return loss_per_epoch_train, train_epoch_loss
        

def val(epoch_num, best_loss, val_loss_per_epoch):
        retinanet.eval()
        loss_per_epoch_val = []
        for iter_num, data in tqdm(enumerate(dataloader_val)):
                with torch.no_grad():
                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                        
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()

                    loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue


                

                val_loss_per_epoch+=loss.item()
                loss_per_epoch_val.append(float(loss))

                if iter_num%print_every==0:
                    print('Epoch:{} | Batch: {} | cls_loss: {:.4f} | reg_loss:{:.4f} | running loss:{:.4f}'.format(epoch_num, iter_num, float(classification_loss.item()), float(regression_loss.item()), val_loss_per_epoch/(iter_num+1)))

                del classification_loss
                del regression_loss


        avg_loss = val_loss_per_epoch/len(dataloader_val)
        val_epoch_loss.append(avg_loss)
        if iter_num%print_every==0:
            print('For epoch:{} total val loss:{:.4f}\n'.format(epoch_num, avg_loss))

        writer.add_scalar('validation_loss', avg_loss, epoch_num+1)            
        if avg_loss < best_loss:
            best_loss=avg_loss
            torch.save({'epoch':epoch_num,'model_state_dict':retinanet.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),'scheduler_dict':scheduler.state_dict(),'best_loss':best_loss}, os.path.join(save_path, 'epoch_{}_{:.4f}_state_dict.pt'.format(epoch_num, best_loss)))
            #torch.save(retinanet,os.path.join(save_path, 'epoch_{}_{:.4f}_model.pt'.format(epoch_num, best_loss)))

        return loss_per_epoch_val, val_epoch_loss, best_loss


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_checkpoint_path',default='../weights/modelv1')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_checkpoint_path', type=str)
    parser.add_argument('--prepare_train_val_data', action='store_true')

    #For collab
    # depth=50
    # epochs=100
    # print_every=300
    # save_checkpoint_path='drive/MyDrive/weights/modelv2'
    # batch_size=6
    # resume=False
    # prepare_train_val_data=False
    ###########

    lr=1e-4 
    parser = parser.parse_args()
    batch_size = parser.batch_size

    print_every = parser.print_every
    save_path = parser.save_checkpoint_path
    save_path_train = save_path.replace(save_path.split('/')[-1],save_path.split('/')[-1]+'_train')
    print('Train weights will be saved at ',save_path_train)
    makedir(save_path)
    makedir(save_path_train)
    writer = SummaryWriter(os.path.join('runs',save_path.split('/')[-1]))

    resume_checkpoint = parser.resume
    print('resume checkpoint boolean value ',resume_checkpoint)

    start_epoch=0
    best_loss=float('inf')

    data_csv = '../data/train.csv'
    class_id_to_name_map = class_id_name_mapping(pd.read_csv(data_csv))
    data_dir, pickle_file, save_path_viz = '../data/all_data','../data/bboxes.pkl', '../data/viz'
    train_dir, test_dir = '../data/train_data/','../data/test_data/'

    prepare_train_val_data=parser.prepare_train_val_data
    print('prepare_train_val_data value ',prepare_train_val_data)

    if prepare_train_val_data:
        split_data(data_dir, train_dir, test_dir)

    dataset_train = VinBigDataset(train_dir, pickle_file, class_id_to_name_map, save_path_viz, visualize=False)
    dataset_val = VinBigDataset(test_dir, pickle_file, class_id_to_name_map, save_path_viz, train=False, visualize=False)    

    if resume_checkpoint:
        resume_checkpoint_path=parser.resume_checkpoint_path
        print('Loading checkpoint from :{}....'.format(resume_checkpoint_path))
        retinanet = model.resnet50(num_classes=dataset_train.num_classes())
        state_dict = torch.load(resume_checkpoint_path)
        retinanet.load_state_dict(state_dict['model_state_dict'])
        optimizer = optim.Adam(retinanet.parameters(), lr=lr)
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        scheduler.load_state_dict(state_dict['scheduler_dict'])
        start_epoch = state_dict['epoch']
        best_loss = state_dict['best_loss']
        print('Checkpoint loaded!')



    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, num_workers=2, collate_fn=collate)    
    #pretrained_model_path = '../coco_resnet_50_map_0_335_state_dict.pt'
    

    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size = batch_size, num_workers=2, collate_fn=collate)#, batch_sampler=sampler_val)

    print('Number of classes: ',dataset_train.num_classes())
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, validating=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, validating=True)
        #retinanet.load_state_dict(torch.load(pretrained_model_path))
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')


    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()


    retinanet.training = True

    if not resume_checkpoint:
        optimizer = optim.Adam(retinanet.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataloader_train.dataset)))
    print('Num training batches: {}'.format(len(dataloader_train)))

    print('Num validation images: {}'.format(len(dataloader_val.dataset)))
    print('Num validation batches: {}'.format(len(dataloader_val)))
    print('lr is ',lr)

    

    for epoch_num in range(start_epoch, parser.epochs):

        retinanet.train()
        retinanet.freeze_bn()

        train_epoch_loss = []
        val_epoch_loss = []

        train_loss_per_epoch = 0.0
        val_loss_per_epoch = 0.0
        loss_per_epoch_train, train_epoch_loss = train(epoch_num, train_loss_per_epoch)
        loss_per_epoch_val, val_epoch_loss, best_loss = val(epoch_num, best_loss, val_loss_per_epoch)


        scheduler.step(np.mean(loss_per_epoch_train))

        

    

    torch.save(retinanet, 'model_final.pt')



