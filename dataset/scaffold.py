import os.path as osp
import torch 
from torch_geometric.data import InMemoryDataset, Data
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from .get_datasets import get_dataset
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rdkit.Chem import DataStructs
import operator
from rdkit.ML.Cluster import Butina
import numpy as np


### Define the dataset class
class ogbg_with_smiles(InMemoryDataset):
    
    """

    A dataset class for handling OGB datasets with SMILES representations.

    Parameters:
        name (str): The name of the dataset.
        root (str): Root directory where the dataset should be saved.
        data_list (list): List of data points.
        clustering_params (dict, optional): Parameters related to clustering, including:
            - scaff_cluster (str): The clustering method, 'k-mean' or 'butina'.
            - smile_list (list): List of SMILES strings corresponding to `data_list`.
            - pca_dim (int): Number of principal components for PCA.
            - n_clusters (int): Number of clusters.
            - cutoff (float): Cutoff distance for clustering.
            - radius (int): Radius parameter for clustering.
            - nBits (int): Number of bits for clustering.
        transform (callable, optional): A function/transform that takes a data object and returns a transformed version.
        pre_transform (callable, optional): A function/transform that is called before saving the dataset to disk.
    """
    
    def __init__(self, name, root, data_list, smile_list = None, meta_dict = None,
                 clustering_params=None, transform=None, pre_transform=None):
        super(ogbg_with_smiles, self).__init__(None, transform, pre_transform)
        self.root = root
        self.smile_list = smile_list
        self.total_data_len = len(data_list) 
        self.data_list = data_list
        self.meta_dict = meta_dict

        print(clustering_params)
        # add smiles and scaffold smiles to data_list
        if smile_list is not None:
            _ , self.scaffold_sets = generate_scaffolds_dict(smile_list)
            for i in range(len(data_list)):
                data_list[i].smiles = smile_list[i]
                scaff_smiles = _generate_scaffold(smile_list[i])
                data_list[i].scaff_smiles = scaff_smiles

            if clustering_params is not None:
                cluster_method = clustering_params['cluster_method']
                pca_dim = clustering_params['pca_dim']
                n_clusters = clustering_params['n_clusters']
                cutoff = clustering_params['cutoff']
                radius = clustering_params['radius']
                nBits = clustering_params['nBits']    
            
                if cluster_method not in ['k-mean', 'butina']:
                    raise ValueError('Clustering method must be either k-mean or butina')
            
                else:
                    scaff_smiles_list = [data.scaff_smiles for data in data_list]
                    cluster_labels = assign_scaff_cluster(scaff_smiles_list, method=cluster_method, n_clusters=n_clusters, pca_dim=pca_dim,
                                                        cutoff=cutoff, radius=radius, nBits=nBits)
                    for i in range(len(data_list)):
                        data_list[i].cluster_id = cluster_labels[i]
        
        self.data, self.slices = self.collate(data_list)
                     
        
        self.name = '_'.join(name.split('-'))
        self.smile_list = smile_list
    
    def get_metadata(self):
        if self.meta_dict is None:
            raise ValueError('No metadata found')
        
        return self.meta_dict
    
    def get_cluster_info(self):
        # output value counts of the cluster labels
        return pd.Series([data.cluster_id for data in self.data_list]).value_counts()
    
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


def assign_scaff_cluster(scaff_list, method = 'k-mean', n_clusters = 10, pca_dim = 3,
                         cutoff = 0.6, radius = 4, nBits = 1024):
    scaff_mols = [Chem.MolFromSmiles(scaffold) for scaffold in scaff_list]
    if method == 'k-mean':
        ecfp = []
        error = []
        for mol in scaff_mols:
            if mol is None:
                print('Error: None molecule')
                error.append(mol)
                ecfp.append([None]*1024)
            else:
                mol = Chem.AddHs(mol)
                list_bits_fingerprint = []
                list_bits_fingerprint[:0] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
                ecfp.append(list_bits_fingerprint)
        ecfp_df = pd.DataFrame(data = ecfp, index = scaff_list)
        
        # reduce the dimension to 3 using PCA
        pca = PCA(n_components = pca_dim, random_state=0)
        ecfp_df_pca = pca.fit_transform(ecfp_df) # numpy array        
        
        # apply k-mean clustering algorithm
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(ecfp_df_pca)

        # get the cluster labels for each data point
        cluster_labels = kmeans.labels_
        if len(cluster_labels) != len(scaff_mols):
            raise ValueError('The number of cluster labels is not equal to the number of scaffold molecules')
        print(f"K-mean Assigned {len(set(scaff_mols))} scaffolds to {n_clusters} clusters")
        return cluster_labels.tolist()
        
    elif method == 'butina':
        ecfp = [] 
        for mol in scaff_mols:
            if mol is None:
                ecfp.append(None)
                continue
            try:
                mol = Chem.AddHs(mol)
                ecfp.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
            except Exception as e:
                print(f"Error generating Morgan fingerprint: {e}")
                ecfp.append(None)
        # calculate distance matrix 
        dists = []
        n_mols  = len(scaff_mols)
        
        for i in range(1, n_mols):
            dist = DataStructs.cDataStructs.BulkTanimotoSimilarity(ecfp[i], ecfp[:i], returnDistance=True)
            dists.extend([x for x in dist])
        
        
        cluster_indices = Butina.ClusterData(dists, n_mols, cutoff, isDistData=True)
        cluster_labels = [-1] * len(ecfp)

        # Iterate over cluster_indices to assign cluster labels
        for cluster_label, cluster in enumerate(cluster_indices):
            for index in cluster:
                cluster_labels[index] = cluster_label

        print(f'Butina Assigned {len(scaff_mols)} molecules to {len(cluster_indices)} clusters')
        return cluster_labels
       
    else:
        raise ValueError('Clustering method must be either k-mean or butina')


def get_tanimoto_similarity_matrix(smile_list, radius=4, nBits=1024):
    smile_mols = [Chem.MolFromSmiles(scaffold) for scaffold in smile_list]
    ecfp = []
    for mol in smile_mols:
        if mol is None:
            ecfp.append(None)
            continue
        try:
            mol = Chem.AddHs(mol)
            ecfp.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))
        except Exception as e:
            print(f"Error generating Morgan fingerprint: {e}")
            ecfp.append(None)

    n_mols = len(smile_mols)
    sim_matrix = np.zeros((n_mols, n_mols))  # Initialize similarity matrix

    for i in range(n_mols):
        for j in range(i+1, n_mols):  # Only compute upper triangle
            if ecfp[i] is None or ecfp[j] is None:
                sim_matrix[i, j] = np.nan
                sim_matrix[j, i] = np.nan
            else:
                sim = DataStructs.TanimotoSimilarity(ecfp[i], ecfp[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric matrix

    for i in range(n_mols):  # Fill the diagonal with 1s for self-similarity
        sim_matrix[i, i] = 1.0

    return sim_matrix
