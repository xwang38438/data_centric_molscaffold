import os.path as osp
import torch 
from torch_geometric.data import InMemoryDataset, Data
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from .get_datasets import get_dataset

class ogbg_with_smiles(InMemoryDataset):
    def __init__(self, name, root, data_list, smile_list = None, 
                 transform=None, pre_transform=None):
        super(ogbg_with_smiles, self).__init__(None, transform, pre_transform)
        self.root = root
        self.smile_list = smile_list
        self.total_data_len = len(data_list) 
        self.data_list = data_list
        _ , self.scaffold_sets = generate_scaffolds_dict(smile_list)
        
        if smile_list is not None:
            for i in range(len(data_list)):
                data_list[i].smiles = smile_list[i]
                scaff_smiles = _generate_scaffold(smile_list[i])
                data_list[i].scaff_smiles = scaff_smiles
  
        
        self.data, self.slices = self.collate(data_list)
        
        # def len(self):
        #     return len(self.data)               
        
        self.name = '_'.join(name.split('-'))
        self.smile_list = smile_list
        
    def get_scaffold_sets(self):
        return self.scaffold_sets

    def get_smile_list(self):
        return self.smile_list
    
    def get_y_list(self):
        return [data.y.item() for data in self.data_list]

    def get_idx_split(self, split_type = 'random', ratio = [0.8, 0.1, 0.1]):
        
        path = osp.join(self.root, self.name, 'split', split_type)
        # create directory if not exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        if split_type not in ['random', 'scaffold']:
            raise ValueError('split_type must be either random or scaffold')
        
        train_ratio, valid_ratio, test_ratio = ratio
        
        if split_type == 'random':
            try:
                train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
                valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
                test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]
            except:
                print(f'Splitting with random seed 42 and ratio{ratio}')
                full_idx = list(range(self.total_data_len))

                train_idx, test_idx, _, _ = train_test_split(full_idx, full_idx, test_size=test_ratio, random_state=42)
                train_idx, valid_idx, _, _ = train_test_split(train_idx, train_idx, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
                df_train = pd.DataFrame({'train': train_idx})
                df_valid = pd.DataFrame({'valid': valid_idx})
                df_test = pd.DataFrame({'test': test_idx})
                df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
                df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
                df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")
            return {'train': torch.tensor(train_idx, dtype = torch.long), 
                    'valid': torch.tensor(valid_idx, dtype = torch.long), 
                    'test': torch.tensor(test_idx, dtype = torch.long)}
                    
        if split_type == 'scaffold':
            # check if data points have smiles attribute
            if not hasattr(self.data_list[0], 'smiles'):
                raise ValueError('Data points do not have smiles attribute')

            try: 
                train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
                valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
                test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]
            except:
                raise ValueError('No split found. Check the ogbg dataset download')

            return {'train': torch.tensor(train_idx, dtype = torch.long), 
                    'valid': torch.tensor(valid_idx, dtype = torch.long), 
                    'test': torch.tensor(test_idx, dtype = torch.long)}
                

def generate_scaffolds_dict(smile_list):
    scaffolds = {}
    data_len = len(smile_list)
    for i in range(data_len):
        scaffold = _generate_scaffold(smile_list[i])
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [i]
            else:
                scaffolds[scaffold].append(i)

    # sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    
    all_scaffold_sets = [
    scaffold_set for (scaffold, scaffold_set) in sorted(scaffolds.items(),
                                                        key = lambda x: (len(x[1]), x[1][0]),
                                                        reverse=True)
    ]
    
    all_scaffold_sets_smiles = [
    (scaffold, scaffold_set) for (scaffold, scaffold_set) in sorted(scaffolds.items(),
                                                        key = lambda x: (len(x[1]), x[1][0]),
                                                        reverse=True)
    ]
    
    
    return all_scaffold_sets, all_scaffold_sets_smiles


from typing import Union
def _generate_scaffold(smiles: str,
                       include_chirality: bool = False) -> Union[str, None]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    if scaffold == '':
        return smiles 

    return scaffold



def get_scaffold_split_info(args):
    labeled_dataset = get_dataset(args, './raw_data')
    labeled_dataset = [data for data in labeled_dataset]
    
    # extract smile list 
    smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
    # unzip the smile file
    smiles = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
    smiles = smiles['smiles'].tolist()
    
    test = ogbg_with_smiles(name = args.dataset,
                                   root = './raw_data',
                                   data_list = labeled_dataset, 
                                   smile_list = smiles)
    
    scaffold_set = test.scaffold_sets
    
    split_idx = test.get_idx_split(split_type='scaffold')
    train_idx = split_idx['train'].tolist()
    valid_idx = split_idx['valid'].tolist()
    test_idx = split_idx['test'].tolist()

    num_train_scaffolds = 0
    num_valid_scaffolds = 0
    num_test_scaffolds = 0

    last_train_idx = train_idx[-1]
    last_valid_idx = valid_idx[-1]
    last_test_idx = test_idx[-1]

    # Initialize flags to check if train, valid, test indices are found
    found_train, found_valid, found_test = False, False, False

    for i, tup in enumerate(scaffold_set):
        # Check if the last train index is in the current scaffold tuple
        # and we haven't found the train index yet
        if not found_train and last_train_idx in tup[1]:
            num_train_scaffolds = i + 1  # +1 because enumeration starts at 0
            print(f'The training set has {len(train_idx)} molecules with {num_train_scaffolds} scaffolds')
            found_train = True

        # Check if the last valid index is in the current scaffold tuple
        # and we haven't found the valid index yet
        if not found_valid and last_valid_idx in tup[1]:
            num_valid_scaffolds = i + 1 - num_train_scaffolds
            print(f'The valid set has {len(valid_idx)} molecules with {num_valid_scaffolds} scaffolds')
            found_valid = True

        # Check if the last test index is in the current scaffold tuple
        # and we haven't found the test index yet
        if not found_test and last_test_idx in tup[1]:
            num_test_scaffolds = i + 1 - num_train_scaffolds - num_valid_scaffolds
            print(f'The test set has {len(test_idx)} molecules {num_test_scaffolds} scaffolds')
            found_test = True

        # Break the loop if all indices are found
        if found_train and found_valid and found_test:
            break


                # all_scaffold_sets, _ = generate_scaffolds_dict(self.smile_list)
                
                # print(f'splitting by scaffold with ratio {ratio}')

                # train_cutoff = train_ratio * len(self.data_list)
                # valid_cutoff = (train_ratio + valid_ratio) * len(self.data_list)
                
                # train_idx: List[int] = []
                # valid_idx: List[int] = []
                # test_idx: List[int] = []

                # for scaffold_set in all_scaffold_sets:
                #     if len(train_idx) + len(scaffold_set) > train_cutoff:
                #         if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                #             test_idx.extend(scaffold_set)
                #         else: 
                #             valid_idx.extend(scaffold_set)
                #     else:
                #         train_idx.extend(scaffold_set)

                # assert len(set(train_idx).intersection(set(valid_idx))) == 0
                # assert len(set(test_idx).intersection(set(valid_idx))) == 0
                
                # df_train = pd.DataFrame({'train': train_idx})
                # df_valid = pd.DataFrame({'valid': valid_idx})
                # df_test = pd.DataFrame({'test': test_idx})
                # df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
                # df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
                # df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")