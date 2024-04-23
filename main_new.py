import math
import time
import pickle
import logging
from tqdm import tqdm
from datetime import datetime
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
import os
from models.gnn import GNN
from configures.arguments import load_arguments_from_yaml, get_args
from dataset.get_datasets import get_dataset
from utils import AverageMeter, validate, print_info, init_weights, load_generator, ImbalancedSampler
from utils import build_augmentation_dataset
from dataset.scaffold import ogbg_with_smiles
import pandas as pd
import gzip

# suppress warnings
warnings.filterwarnings("ignore")

cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
reg_criterion = torch.nn.MSELoss(reduction='none')

def get_logger(name, logfile=None):
    """ create a nice logger """
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger

def seed_torch(seed=0):
    print('Seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    # num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(1e-2, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(args, model, train_loaders, optimizer, scheduler, epoch):
    if args.task_type in 'regression':
        criterion = reg_criterion
    else:
        criterion = cls_criterion
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = args.device
    model.train()
    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            batch_labeled = next(train_loaders['labeled_iter'])
        except:
            train_loaders['labeled_iter'] = iter(train_loaders['labeled_trainloader'])
            batch_labeled = next(train_loaders['labeled_iter'])
        batch_labeled =  batch_labeled.to(device)
        targets = batch_labeled.y.to(torch.float32)
        is_labeled = targets == targets
        if batch_labeled.x.shape[0] == 1 or batch_labeled.batch[-1] == 0:
            continue
        else:            
            pred_labeled = model(batch_labeled)[0]
            Losses = criterion(pred_labeled.view(targets.size()).to(torch.float32)[is_labeled], targets[is_labeled])
            loss = Losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.8f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.steps,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                ))
            p_bar.update()
    if not args.no_print:
        p_bar.close()
    return train_loaders

def main(args):    
    device = torch.device('cuda', args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
    smiles = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
    smiles = smiles['smiles'].tolist()  
    labeled_dataset = get_dataset(args, './raw_data')
    labeled_dataset_list = [data for data in labeled_dataset]
    
    # define clustering parameters
    meta_dict = {
        'num_tasks': labeled_dataset.num_tasks,
        'eval_metric': labeled_dataset.eval_metric,
        'task_type': labeled_dataset.task_type,
        'num_classes': labeled_dataset.__num_classes__,
        'binary': labeled_dataset.binary,
    }
    
    cluster_dict = {
        'cluster_method': args.cluster_method,
         'pca_dim': args.pca_dim,
         'n_clusters': args.n_clusters,
         'cutoff': args.cut_off,
         'radius': args.radius,
         'nBits': args.n_bits        
    }
    
    new_labeled_dataset = ogbg_with_smiles(name = args.dataset,
                                   root = './raw_data',
                                   data_list = labeled_dataset_list, 
                                   smile_list = smiles,
                                   clustering_params=cluster_dict,
                                   meta_dict=meta_dict)
    
    if args.split == 'scaffold':
        label_split_idx = new_labeled_dataset.get_idx_split(split_type = 'scaffold')
    else:
        label_split_idx = new_labeled_dataset.get_idx_split()
    
    
    args.num_trained = len(label_split_idx["train"])
    args.num_trained_init = args.num_trained
    args.task_type = labeled_dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1
    args.strategy = args.strategy_init

    if args.dataset == 'ogbg-molhiv':
        sampler = ImbalancedSampler(new_labeled_dataset, label_split_idx["train"])
        labeled_trainloader = DataLoader(new_labeled_dataset[label_split_idx["train"]], batch_size=args.batch_size, sampler=sampler, num_workers = args.num_workers)        
    else:
        labeled_trainloader = DataLoader(new_labeled_dataset[label_split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    valid_loader = DataLoader(new_labeled_dataset[label_split_idx["valid"]], batch_size=args.batch_size, shuffle=False,num_workers = args.num_workers)
    test_loader = DataLoader(new_labeled_dataset[label_split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    model = GNN(gnn_type = args.model, num_tasks = labeled_dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, 
                drop_ratio = args.drop_ratio, graph_pooling = args.readout, norm_layer = args.norm_layer).to(device)
    
    generator = load_generator(device, path='checkpoints/pcba_denoise.pth')
    init_weights(model, args.initw_name, init_gain=0.02)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 100)
    logging.warning( f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_trained}/{len(label_split_idx['valid'])}/{len(label_split_idx['test'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")
    train_loaders = {'labeled_iter': iter(labeled_trainloader),'labeled_trainloader': labeled_trainloader} 
    
    # topk_mols_dict = {}
    aug_label_dist = []   
    for epoch in range(0, args.epochs):
        train_loaders = train(args, model, train_loaders, optimizer, scheduler, epoch)
        train_perf = validate(args, model, labeled_trainloader)
        valid_perf = validate(args, model, valid_loader)

        if epoch >= args.start and epoch % args.iteration == 0 and epoch < args.end:            
            # need to update the labeled dataset
            print('start one round of augmentation')
            new_dataset, topk_mols = build_augmentation_dataset(args, model, generator, new_labeled_dataset, split=args.split)
            print('end one round of augmentation')

            # topk_mols_dict[epoch] = topk_mols
            aug_label_dist.extend([mol.y for mol in topk_mols])
            
            if args.dataset == 'ogbg-molhiv':  # may apply to all imbalanced datasets
                sampler = ImbalancedSampler(new_dataset, new_dataset.get_idx_split()["train"])
                new_trainloader = DataLoader(new_dataset, batch_size=args.batch_size, sampler=sampler,num_workers = args.num_workers)        
            else:
                new_trainloader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
            train_loaders['labeled_trainloader'] = new_trainloader
            args.num_trained = len(new_trainloader.dataset)
            args.steps = args.num_trained // args.batch_size + 1
            if args.strategy.split('_')[-1] == 'accumulate':
                new_labeled_dataset = new_dataset
            if len(new_trainloader.dataset) > args.num_trained_init * 2:
                args.strategy = 'replace' + '_' + args.strategy.split('_')[-1]

        update_test = False
        if epoch != 0 and 'classification' in args.task_type and valid_perf['auc'] >  best_valid_perf['auc']:
            update_test = True
        elif epoch != 0 and 'regression' in args.task_type and valid_perf['mae'] <  best_valid_perf['mae']:
            update_test = True
        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            best_train_perf = train_perf
            cnt_wait = 0
            best_epoch = epoch
            test_perf = validate(args, model, test_loader)
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                print_info('Test', test_perf)
        else:
            # not update
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
            if epoch > 30: 
                cnt_wait += 1
                if cnt_wait > args.patience:
                    break

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print_info('train', best_train_perf)
    print_info('valid', best_valid_perf)
    print_info('test', test_perf)
    # save topk_mols_dict
    if args.get_topk_mols:
    
        os.makedirs(f'./figures/label_imbalance/', exist_ok=True)
        torch.save(aug_label_dist, f'./figures/label_imbalance/{args.dataset}_{args.model}_nc{args.n_clusters}_topk{args.topk}.pt')

        # Save the dictionary as a pickle file
        # with gzip.open(f'./results/{args.dataset}/{args.split}_{args.model}_topk_mols.pkl.gz', 'wb') as f:
        #     pickle.dump(topk_mols_dict, f)
        
    return best_train_perf, best_valid_perf, test_perf

if __name__ == '__main__':
    args = get_args()

    config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
    for arg, value in config.items():
        setattr(args, arg, value)
    
    args.strategy_init = args.strategy
    datetime_now = datetime.now().strftime("%Y%m%d.%H%M%S")
    logger = get_logger(__name__, logfile=None)
    print(args)
    results = {}
    for exp_num in range(args.trails):
        seed_torch(exp_num)
        args.exp_num = exp_num
        train_perf, valid_perf, test_perf = main(args)
        exp_result_temp = {'train': train_perf, 'valid': valid_perf, 'test': test_perf}
        if exp_num == 0:
            for metric in train_perf.keys():
                results[f'train_{metric}'] = []
                results[f'valid_{metric}'] = []
                results[f'test_{metric}'] = []
        for name in ['train', 'test', 'valid']:
            if args.task_type in 'regression':
                metric_list = ['rmse', 'r2','mae','mse']
            else:
                metric_list = ['auc']
            for metric in metric_list:
                results[f'{name}_{metric}'].append(exp_result_temp[name][metric])
        for mode, nums in results.items():
            print('{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums))
            
        # save results
    results_dir = f'./results/{args.dataset}'
    # calculate the mean and std of the results
    os.makedirs(results_dir, exist_ok=True)
        
    results_df = pd.DataFrame(results)
    # add new columns for mean and std of test results
    for metric in results_df.columns:
        if metric.startswith('test'):
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            # add new columns and assign the mean and std
            results_df[f'{metric}_mean'] = mean
            results_df[f'{metric}_std'] = std
    
    
    results_df.to_csv(f'{results_dir}/{args.model}_{args.cluster_method}_{args.n_clusters}_top{args.topk}_{args.strategy}_nn{args.n_negative}.csv', index=False)