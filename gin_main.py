import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gin_vanilla.gnn import GNN
import os 
from dataset.scaffold import ogbg_with_smiles
import pandas as pd
from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

test_cls_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
test_reg_criterion = torch.nn.MSELoss(reduction='none')
### to do 
# load dataframe for the tsne visualization



def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator, task_type):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    #print(y_true.shape, y_pred.shape)
    
    if 'classification' in task_type:
            #loss = test_cls_criterion(torch.tensor(y_pred).to(torch.float32), torch.tensor(y_true).to(torch.float32))
        # Calculate loss for each task separately
        y_pred_tensor = torch.tensor(y_pred).to(torch.float32)
        y_true_tensor = torch.tensor(y_true).to(torch.float32)
        # print(y_pred_tensor.shape, y_true_tensor.shape)
        is_labeled = y_true_tensor == y_true_tensor
        # print(is_labeled.shape )
        # Calculate loss for each task and each test point separately
        loss = test_cls_criterion(y_pred_tensor, y_true_tensor)
        # print(loss.shape)
        
        loss = loss.detach().cpu().numpy()
        # check if there exists nan target
        # print(np.isnan(loss).any())
        loss = np.nanmean(loss, axis=1)

        print(f'Check if loss has nan values: {np.isnan(loss).any()}')
    else:
        loss = test_reg_criterion(torch.tensor(y_pred).to(torch.float32), torch.tensor(y_true).to(torch.float32))
    

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root='./raw_data')
    dataset_list = [data for data in dataset]
    smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
    smiles_df = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
    smiles_list = smiles_df['smiles'].tolist()

    new_dataset = ogbg_with_smiles(name = args.dataset,
                                       root = './raw_data',
                                       data_list = dataset_list, 
                                       smile_list = smiles_list)

    # need to collect num_tasks, eval_metric, and task_type from the dataset


    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        new_dataset.data.x = new_dataset.data.x[:,:2]
        new_dataset.data.edge_attr = new_dataset.data.edge_attr[:,:2]

    split_idx = new_dataset.get_idx_split(split_type="scaffold")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    # need change to the newly defined dataset
    train_loader = DataLoader(new_dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(new_dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(new_dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    test_losses = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf, _ = eval(model, device, train_loader, evaluator, dataset.task_type)
        valid_perf, _ = eval(model, device, valid_loader, evaluator, dataset.task_type)
        test_perf, test_loss = eval(model, device, test_loader, evaluator, dataset.task_type)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        test_losses.append(test_loss)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
        best_test_loss = test_losses[best_val_epoch]
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)
        best_test_loss = test_losses[best_val_epoch]

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

    # save the best test loss list
    # if folder does not exist, create it
    if not os.path.exists(f'./results/{args.dataset}'):
        os.makedirs(f'./results/{args.dataset}')
    torch.save(best_test_loss, f'./results/{args.dataset}/test_losses.pt')


if __name__ == "__main__":
    main()