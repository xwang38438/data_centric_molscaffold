import torch
from dataset.get_datasets import get_dataset
import warnings
import numpy as np
from dataset.scaffold import ogbg_with_smiles
import os 
import operator
import gzip
from rdkit.ML.Cluster import Butina
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
import plotter.descriptors as desc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
sys.path.append('./plotter/')
from plotter.plot import convert_idx_list, Orig_Plotter
from dataset.scaffold import get_scaffold_split_info, _generate_scaffold, generate_scaffolds_dict
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist, squareform


warnings.filterwarnings('ignore')

def get_ecfp_fingerprints(smiles_list, radius=4, nBits=1024):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES 
    :param radius: The ECPF fingerprints radius.
    :param nBits: The number of bits of the fingerprint vector.
    :type radius: int
    :type smiles_list: list
    :type nBits: int
    :returns: The calculated ECPF fingerprints for the given SMILES
    """  
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Scaffold-aware pipeline to cluster molecules')
    parser.add_argument('--dataset', type=str, default='ogbg-molbace',
                        help='dataset name (default: ogbg-molbace)')
    
    parser.add_argument('--c_method', type=str, default='k-mean',
                        help='clustering method (k-mean or butina)')
    parser.add_argument('--num_clusters', type=int, default=30,
                        help='number of clusters to separate scaffold structures(default: 10)')
    parser.add_argument('--pca_dim', type=int, default=8,
                        help='dimensionality of PCA (default: 3)')
    
    
    args = parser.parse_args()
    print(args.dataset)
    
    labeled_dataset = get_dataset(args, './raw_data')
    labeled_dataset_list = [data for data in labeled_dataset]
    smile_path = os.path.join('./raw_data', '_'.join(args.dataset.split('-')), 'mapping/mol.csv.gz')
    smiles_df = pd.read_csv(smile_path, compression='gzip', usecols=['smiles'])
    smiles = smiles_df['smiles'].tolist() 

    meta_dict = {
        'num_tasks': labeled_dataset.num_tasks,
        'eval_metric': labeled_dataset.eval_metric,
        'task_type': labeled_dataset.task_type,
        'num_classes': labeled_dataset.__num_classes__,
        'binary': labeled_dataset.binary,
    }
    
    cluster_dict = {
        'cluster_method': 'k-mean',
         'pca_dim': 3,
         'n_clusters': 25,
         'cutoff': 0.8,
         'radius': 4,
         'nBits': 1024      
        
    }

  
    new_labeled_dataset = ogbg_with_smiles(name = args.dataset,
                                   root = './raw_data',
                                   data_list = labeled_dataset_list, 
                                   smile_list = smiles,
                                   clustering_params=cluster_dict,
                                   meta_dict=meta_dict)
    

    # get molecules indices for train, valid, and test split 
    label_split_idx_scaffold = new_labeled_dataset.get_idx_split(split_type = 'scaffold')
    mol = new_labeled_dataset[0]
    print(mol)
    mol.test = True
    print(mol)
    print(labeled_dataset[0])
    
    # print(new_labeled_dataset.get_metadata())
    # print(new_labeled_dataset.get_cluster_info())
    
    

    
    
    # # derive the scaffold smiles for all molecules
    # scaffold_list = new_labeled_dataset.scaff_smiles 
    # scaffold_mols = [Chem.MolFromSmiles(scaffold) for scaffold in scaffold_list]
    # print(len(scaffold_mols))

    # if args.c_method == 'k-mean':
    #     ecfp = []
    #     error = []
    #     for mol in scaffold_mols:
    #         if mol is None:
    #             error.append(mol)
    #             ecfp.append([None]*1024)
    #         else:
    #             mol = Chem.AddHs(mol)
    #             list_bits_fingerprint = []
    #             list_bits_fingerprint[:0] = AllChem.GetMorganFingerprint(mol, 4, 1024)
    #             ecfp.append(list_bits_fingerprint)
    #     ecfp_df = pd.DataFrame(data = ecfp, index = scaffold_list)
        
    #     # reduce the dimension to 3 using PCA
    #     pca = PCA(n_components = args.pca_dim, random_state=0)
    #     ecfp_df_pca = pca.fit_transform(ecfp_df) # numpy array        
           
    #     # apply k-mean clustering algorithm
    #     kmeans = KMeans(n_clusters=args.num_clusters, random_state=0)
    #     kmeans.fit(ecfp_df_pca)

    #     # get the cluster labels for each data point
    #     cluster_labels = kmeans.labels_
    #     if len(cluster_labels) != len(scaffold_mols):
    #         raise ValueError('The number of cluster labels is not equal to the number of scaffold molecules')
        
    #     # assign the cluster labels to dataset
    #     new_labeled_dataset.scaff_labels = cluster_labels
    #     print(pd.Series(cluster_labels).value_counts())
        
    #     # print the cluster labels
    #     print(new_labeled_dataset.scaff_labels)
    #     print(len(cluster_labels))
        
    # elif args.c_method == 'butina':
    #     fps = [] # list of rdkit.DataStructs.cDataStructs.ExplicitBitVect objects
    #     for mol in scaffold_mols:
    #         mol = Chem.AddHs(mol)
    #         fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 4, 1024))
        
    #     # calculate distance matrix 
        
        
    #     dists = []
    #     n_mols  = len(scaffold_mols)
        
    #     for i in range(1, n_mols):
    #         dist = DataStructs.cDataStructs.BulkTanimotoSimilarity(fps[i], fps[:i], returnDistance=True)
    #         dists.extend([x for x in dist])
        
        
    #     cutoff = 0.65
        
    #     cluster_indices = Butina.ClusterData(dists, n_mols, cutoff, isDistData=True)

    #     labels = [-1] * len(fps)

    #     # Iterate over cluster_indices to assign cluster labels
    #     for cluster_label, cluster in enumerate(cluster_indices):
    #         for index in cluster:
    #             labels[index] = cluster_label

    #     # Now, labels list contains the cluster label for each molecule in the order they appear in fps
    #     print(labels)

    #     cluster_mols = [operator.itemgetter(*cluster)(scaffold_mols) for cluster in cluster_indices]
        
    #     print(len(cluster_mols))
    #     print(cluster_mols[0][:5])
    #     print(type(cluster_mols[0]))
        
    # # elif args.c_method == 'k-medoids':
    # #     fps = [] # list of rdkit.DataStructs.cDataStructs.ExplicitBitVect objects
    # #     for mol in scaffold_mols:
    # #         mol = Chem.AddHs(mol)
    # #         fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 4, 1024))


    # #     fps_df = pd.DataFrame([list(fp) for fp in fps])
    # #     # Compute the Tanimoto distance matrix
    # #     distances = pdist(fps_df.values, metric='jaccard')
    # #     dist_matrix = squareform(distances)
    # #     # Create a KMedoids instance with the desired number of clusters
    # #     kmedoids = KMedoids(n_clusters = 2, metric='precomputed')
    # #     kmedoids.fit(dist_matrix)
    # #     clusters = kmedoids.labels_
    # #     # value counts of the clusters
    # #     print(pd.Series(clusters).value_counts())
        
        

    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
