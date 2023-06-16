import argparse
import json
import random
import shutil
import os
from tqdm import tqdm
import numpy as np
import time
import gc

from MuserDataset import MuserDataset
from MuserFormatter import MuserFormatter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
from torch.autograd import Variable

from transformers import AdamW

from MuserPLM import MuserPLM


parser = argparse.ArgumentParser(description='Lawformer MUSER')
parser.add_argument('--data',
                    metavar='DIR',
                    default='/data2/private/liqingquan/mjjd_simcase/plm_retrieval/',
                    help='path to dataset')
parser.add_argument('--model_path',
                    metavar='DIR',
                    default='thunlp/Lawformer',
                    help='path to model')
parser.add_argument('--output_path',
                    metavar='DIR',
                    default='/data2/private/liqingquan/mjjd_simcase/plm_retrieval/checkpoints/',
                    help='path to model')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=10,
                    type=int,
                    metavar='N',
                    help='manual epoch number')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=16,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total batch size of all GPUs on the current '
                         'node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-5,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd',
                    '--weight_decay',
                    default=.0,
                    type=float,
                    metavar='W',
                    help='weight decay (default: .0)',
                    dest='wd')
parser.add_argument('--max_seq_len',
                    default=1203,
                    type=int,
                    metavar='M',
                    help='max sequence length')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--main_rank',
                    default=-1,
                    type=int,
                    help='main rank for distributed training')
parser.add_argument('--seed',
                    default=233,
                    type=int,
                    help='seed for initializing training')
parser.add_argument('--global_steps',
                    default=0,
                    type=int,
                    help='Training steps')
parser.add_argument('--checkpoint_steps',
                    default=1,
                    type=int,
                    help='Numbers of steps to save a checkpoint')
parser.add_argument('--test_steps',
                    default=1,
                    type=int,
                    help='Numbers of epoch steps to test')
parser.add_argument('--gpu',
                    type=str,
                    default='0',
                    help="gpu choose, eg. '0,1,2,...' ")


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def gather(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.tolist()


def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'total': 0}
    pred = torch.max(logit, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((pred == label).sum())
    return acc_result


def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpu = len(args.gpu.split(','))
    gpus = [ _ for _ in range(n_gpu) ]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    dist.init_process_group(backend='nccl',
                            init_method='env://')

    train_log = {'parameters': {'epochs': args.epochs,
                                'batch_size': args.batch_size,
                                'learning_rate': args.lr,
                                'weight_decay': args.wd},
                 'training_states': []}

    # create model
    model = MuserPLM(args.model_path)

    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.wd)
        
    main_worker(args.local_rank, args.nprocs, args, train_log, model, optimizer, gpus)


def main_worker(local_rank, nprocs, args, train_log, model, optimizer, gpus):
    batch_size = int(args.batch_size / nprocs)

    # load dataset
    train_dataset = MuserDataset(os.path.join(args.data, 'train.json'), 'train')
    valid_dataset = MuserDataset(os.path.join(args.data, 'test.json'), 'valid')
    train_formatter = MuserFormatter(args.model_path, 'train')
    valid_formatter = MuserFormatter(args.model_path, 'valid')

    def train_collate_fn(data):
        return train_formatter.process(data, "train")

    def valid_collate_fn(data):
        return valid_formatter.process(data, "valid")

    train_simpler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=args.workers,
                                                   collate_fn=train_collate_fn,
                                                   pin_memory=True,
                                                   sampler=train_simpler)
    valid_simpler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size * 2,
                                                   num_workers=args.workers,
                                                   collate_fn=valid_collate_fn,
                                                   pin_memory=True,
                                                   sampler=valid_simpler)
    
    for epoch in range(args.start_epoch, args.epochs):
        train_simpler.set_epoch(epoch)
        valid_simpler.set_epoch(epoch)

        # train for on epoch
        loss = train(train_dataloader, model, optimizer, epoch, local_rank, args)
            
        # evaluate on validation set
        vloss, acc_result = validate(valid_dataloader, model, local_rank, args, epoch)

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.module.state_dict()
                },
                False,
                args,
                f'{epoch}_checkpoint.pth.tar'
            )
                
            train_log['training_states'].append(
                {
                    'epoch': epoch,
                    'train_loss': loss,
                    'valid_loss': vloss,
                    'acc_result': acc_result
                }
            )
        
            with open(args.output_path + f'log.json', 'w') as fout:
                json.dump(train_log, fout, indent=2)
            fout.close()

        torch.distributed.barrier()


def train(train_dataloader, model, optimizer, epoch, local_rank, args):
    model.train()

    total_loss = 0
    acc_result = None

    for step, batch in enumerate(train_dataloader):
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = Variable(batch[key].cuda(non_blocking=True))

        model.zero_grad()
        output = model(data=batch,
                       acc_result=acc_result,
                       mode='train')
        acc_result = output['acc_result']
        loss = output['loss']
        
        torch.distributed.barrier()
        
        reduced_loss = reduce_mean(loss, args.nprocs)
        total_loss += reduced_loss.item()

        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        args.global_steps += 1

    if local_rank == args.main_rank:
        avg_loss = total_loss / len(train_dataloader)
        print(f'Training Loss: {avg_loss:.2f}')
        return avg_loss
    else:
        return 0


def validate(valid_dataloader, model, local_rank, args, epoch):
    model.eval()

    total_loss = 0
    acc_result = None
    res_scores = []

    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = Variable(batch[key].cuda(non_blocking=True))

            output = model(data=batch,
                           acc_result=acc_result,
                           mode='valid')
            loss = output['loss']
            acc_result = output['acc_result']
            total_loss += float(loss)

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            total_loss += reduced_loss.item()
    
    if local_rank == args.main_rank:
        avg_loss = total_loss / len(valid_dataloader)
        print(f'Valid Loss: {avg_loss:.2f}')
        return avg_loss, acc_result
    else:
        return 0, 0


def save_checkpoint(state, is_best, args, file_name='checkpoint.pth.tar'):
    file_name=args.output_path + file_name
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, args.output_path + 'model_best.pth.tar')


if __name__ == '__main__':
    main()

