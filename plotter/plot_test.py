import pandas as pd
from pandas.api.types import is_numeric_dtype
import descriptors as desc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import parameters as parameters
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import numpy as np
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.transform import transform, factor_cmap
from bokeh.palettes import Category10, Inferno, Spectral4
from bokeh.models.mappers import LinearColorMapper
from bokeh.models import ColorBar, HoverTool, Panel, Tabs
from rdkit.Chem import Draw
from io import BytesIO
import base64

class Plotter(object):
    """
    A class used to plot the ECFP fingerprints of the molecules 
    
    :param __mols: list of valid molecules that can be plotted
    :param __target: list containing the target values. Is empty if a target does not exist
    :param tsne_fit: t-SNE object created when the corresponding algorithm is applied to the data
    :param df_plot_xy: dataframe containing the coordinates that have been plotted
    
    :param __df_descriptors: datatframe containing the descriptors representation of each molecule
    :param __df_2_components: dataframe containing the two-dimenstional representation of each molecule
    :param __plot_title: title of the plot reflecting the dimensionality reduction algorithm used
    
    
    :type tsne_fit: sklearn.manifold.TSNE
    :type __plot_title: string
    :type __data: list    
    :type df_plot_xy: Dataframe
    """

    _target_types = {'R', 'C'}
    _sim_types = {'tailored', 'structural'}
    _static_plots = {'scatter', 'hex', 'kde'}
    
    def __init__(self, encoding_list, target, idx, target_type, sim_type, get_desc, get_fingerprints):

        # Error handeling sym_type
        if sim_type not in self._sim_types:
            if len(target) > 0:
                self.__sim_type = 'tailored' 
                print('sim_type indicates the similarity type by which the plots are constructed.\n' +
                      'The supported similarity types are structural and tailored.\n' +
                      'Because a target list has been provided \'tailored\' as been selected as sym_type.')
            else: 
                self.__sim_type = 'structural' 
                print('sim_type indicates the similarity type by which the plots are constructed.\n' +
                      'The supported similarity types are structural and tailored.\n' +
                      'Because no target list has been provided \'structural\' as been selected as sym_type.')
        else:
            self.__sim_type = sim_type
         
        if self.__sim_type != "structural" and len(target) == 0:
            raise Exception("Target values missing")
        
        
        # Error handeling target_type                
        if len(target) > 0:
            if len(target) != len(encoding_list):
                raise Exception("If target is provided its length must match the instances of molecules")
            
        if len(target) > 0:
            df_target = pd.DataFrame(data=target)
            unique_targets_ratio = 1.*df_target.iloc[:, 0].nunique()/df_target.iloc[:, 0].count() < 0.05
            numeric_target = is_numeric_dtype(df_target.dtypes[0])
            if target_type == 'R' and (unique_targets_ratio or not numeric_target):
                print('Input received is \'R\' for target values that seem not continuous.')
            if target_type not in self._target_types:
                if not unique_targets_ratio and numeric_target:
                    self.__target_type = 'R'
                    print('target_type indicates if the target is a continuous variable or a class label.\n'+
                          'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                          'From analysis of the target, R has been selected for target_type.')
                else:
                    self.__target_type = 'C'
                    print('target_type indicates if the target is a continuous variable or a class label.\n'+
                          'R stands for regression and C for classification. Input R as target type for continuous variables and C for class labels.\n'+
                          'From analysis of the target, C has been selected for target_type.')
            else:
                self.__target_type = target_type
        else:
            self.__target_type = None

        if self.__sim_type == "tailored":
            self.__mols, df_descriptors, target = get_desc(encoding_list, target)
            if df_descriptors.empty:
                raise Exception("Descriptors could not be computed for given molecules")
            self.__df_descriptors, self.__target = desc.select_descriptors_lasso(df_descriptors,target,kind=self.__target_type)
        elif self.__sim_type == "structural":
            self.__mols, self.__df_descriptors, self.__target = get_fingerprints(encoding_list,target,2,2048)      
            
        if len(self.__mols) < 2 or len(self.__df_descriptors.columns) < 2:
            raise Exception("Plotter object cannot be instantiated for given molecules")
                
        self.__df_2_components = None
        self.__plot_title = None
        self.__idx = idx
        
        
    @classmethod
    def from_smiles(cls, smiles_list, target = [], idx = [], target_type=None, sim_type=None):
        """
        Class method to construct a Plotter object from a list of SMILES.
        
        :param smile_list: List of the SMILES representation of the molecules to plot.       
        :param target: target values 
        :param target_type: target type R (regression) or C (classificatino)
        :param sim_type: similarity type structural or tailored 
        
        :type smile_list: list
        :type target: list
        :type target_type: string
        :type sim_type: string
        :returns: A Plotter object for the molecules given as input.
        :rtype: Plotter
        
        """
        return cls(smiles_list, target, idx, target_type, sim_type, desc.get_mordred_descriptors, desc.get_ecfp)         
    
    
    def tsne(self, perplexity=None, pca=False, random_state=None, **kwargs):
        """
        Calculates the first 2 t-SNE components of the molecular descriptors.
        
        :param perplexity: perplexity value for the t-SNE model  
        :param pca: indicates if the features must be preprocessed by PCA
        :param random_state: random seed that can be passed as a parameter for reproducing the same results     
        :param kwargs: Other keyword arguments are passed down to sklearn.manifold.TSNE
        :type perplexity: int
        :type pca: boolean
        :type random_state: int
        :type kwargs: key, value mappings
        :returns: The dataframe containing the t-SNE components.
        :rtype: Dataframe
        """ 
        self.__data = self.__data_scaler()
        self.__plot_title = "t-SNE plot"
        
        # Preprocess the data with PCA
        if pca and self.__sim_type == "structural":
            pca = PCA(n_components=10, random_state=random_state)
            self.__data = pca.fit_transform(self.__data)
            self.__plot_title = "t-SNE plot from components with cumulative variance explained " + "{:.0%}".format(sum(pca.explained_variance_ratio_))
        else:
            self.__plot_title = "t-SNE plot"
        
        # Get the perplexity of the model
        if perplexity is None:
            if self.__sim_type == "structural":
                if pca:
                    perplexity = parameters.perplexity_structural_pca(len(self.__data))
                else:
                    perplexity = parameters.perplexity_structural(len(self.__data))
            else:
                perplexity = parameters.perplexity_tailored(len(self.__data))
        else:
            if perplexity<5 or perplexity>50:
                print('Robust results are obtained for values of perplexity between 5 and 50')
        
        # Embed the data in two dimensions
        self.tsne_fit = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, **kwargs)
        ecfp_tsne_embedding = self.tsne_fit.fit_transform(self.__data)
        # Create a dataframe containinting the first 2 TSNE components of ECFP 
        self.__df_2_components = pd.DataFrame(data = ecfp_tsne_embedding
             , columns = ['t-SNE-1', 't-SNE-2'])
            
        if len(self.__target) > 0: 
            self.__df_2_components['target'] = self.__target
        
        if len(self.__idx) > 0:
            self.__df_2_components['idx'] = self.__idx
        
        return self.__df_2_components.copy()
    
    def visualize_plot(self, size=20, kind="scatter", remove_outliers=False, is_colored=True, colorbar=False, clusters=False, filename=None, title=None):
        """
        Generates a plot for the given molecules embedded in two dimensions.
        
        :param size: Size of the plot  
        :param kind: Type of plot 
        :param remove_outliers: Boolean value indicating if the outliers must be identified and removed 
        :param is_colored: Indicates if the points must be colored according to target 
        :param colorbar: Indicates if the plot legend must be represented as a colorbar. Only considered when the target_type is "R".
        :param clusters: If True the clusters are shown instead of possible targets. Pass a list or a int to only show selected clusters (indexed by int).
        :param filename: Indicates the file where to save the plot
        :param title: Title of the plot.
        :type size: int
        :type kind: string
        :type remove_outliers: boolean
        :type is_colored: boolean
        :type colorbar: boolean
        :type clusters: boolean or list or int
        :type filename: string
        :type title: string
        :returns: The matplotlib axes containing the plot.
        :rtype: Axes
        """
        if self.__df_2_components is None:
            print('Reduce the dimensions of your molecules before creating a plot.')
            return None
        
        if clusters is not False and 'clusters' not in self.__df_2_components:
            print('Call cluster() before visualizing a plot with clusters.')
        
        if title is None:
            title = self.__plot_title
        
        if kind not in self._static_plots:
            kind = 'scatter'
            print('kind indicates which type of plot must be visualized. Currently supported static visualization are:\n'+
                  '-scatter plot (scatter)\n'+
                  '-hexagon plot (hex)\n'+
                  '-kernel density estimation plot (kde)\n'+
                  'Please input one between scatter, hex or kde for parameter kind.\n'+
                  'As default scatter has been taken.')
        
        # change this part to include idx and target
        x, y, idx, df_data = self.__parse_dataframe()
        
        # Define colors 
        hue = None
        hue_order = None
        palette = None
        if clusters is not False and 'clusters' in self.__df_2_components.columns:
            hue = 'clusters'
            palette = 'deep'
            if not isinstance(clusters, bool):
                if isinstance(clusters, int): clusters = [clusters]
                df_data['clusters'] = df_data['clusters'].isin(clusters)
                # Labels cluster
                total = df_data['clusters'].value_counts()
                t_s = total.get(True) if total.get(True) else 0
                p_s = t_s / total.sum()
                p_o = 1 - p_s
                labels = {
                    True: f'Selected - {p_s:.0%}', 
                    False: f'Other - {p_o:.0%}'
                    }
                df_data.clusters.replace(labels, inplace=True)
                hue_order = list(labels.values())
            else:
                hue_order = self.__percentage_clusters(df_data)
                hue_order.sort()
        else:
            if len(self.__target) == 0:
                is_colored = False;
            else:
                if is_colored:
                    df_data = df_data.assign(target=self.__target)
                    hue = 'target'
                    palette = 'deep'
                    if self.__target_type == "R":
                        palette = sns.color_palette("inferno", as_cmap=True)
                        
        
        # Remove outliers (using Z-score)
        if remove_outliers:
            df_data = self.__remove_outliers(x, y, df_data)
            
        # Define plot aesthetics parameters
        sns.set_style("dark")
        sns.set_context("notebook", font_scale=size*0.15)
        fig, ax = plt.subplots(figsize=(size,size))
        
        # Create a plot based on the reduced components 
        if kind == "scatter":
            # this part is changed
            marker_styles = {'train': 's', 'valid': '^', 'test': 'o'}
            #plot = plt.figure()

            for split_type, marker in marker_styles.items():
                df_subset = df_data[df_data['idx'] == split_type]
                # add jitter
                df_subset[x] += np.random.normal(scale=0.1, size=len(df_subset))
                df_subset[y] += np.random.normal(scale=0.1, size=len(df_subset))
                plot = sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, 
                                       data=df_subset, s=size*3, marker=marker, legend=False)

            # plot = sns.scatterplot(x=x, y=y, hue=hue, hue_order=hue_order, palette=palette, data=df_data, s=size*3)
            plot.set_label("scatter")
            axis = plot
            plot.legend(markerscale=size*0.19)
            # Add colorbar
            if self.__target_type == "R" and colorbar:
                plot.get_legend().remove()
                norm = plt.Normalize(df_data['target'].min(), df_data['target'].max())
                cm = plt.cm.ScalarMappable(cmap="inferno", norm=norm)
                cm.set_array([])
                plot.figure.colorbar(cm)
        elif kind == "hex":
            plot = ax.hexbin(df_data[x], df_data[y], gridsize=40, cmap='Blues')
            fig.colorbar(plot, ax=ax)
            ax.set_label("hex")
            axis = ax
        elif kind == "kde":
            plot = sns.kdeplot(x=x, y=y, shade=True, data=df_data)
            plot.set_label("kde")
            axis = plot
             
        # Remove units from axis
        axis.set(yticks=[])
        axis.set(xticks=[])
        # Add labels
        axis.set_title(title,fontsize=size*2)
        axis.set_xlabel(x,fontsize=size*2)
        axis.set_ylabel(y,fontsize=size*2)
        
        # Save plot
        if filename is not None:
            fig.savefig(filename)
            
        self.df_plot_xy = df_data[[x,y]]
        
        return axis
    
    def __data_scaler(self):
        # Scale the data
        if self.__sim_type != "structural":
            scaled_data = StandardScaler().fit_transform(self.__df_descriptors.values.tolist())
        else:
            scaled_data = self.__df_descriptors.values.tolist()
            
        return scaled_data
    
    def __parse_dataframe(self):
        x = self.__df_2_components.columns[0]
        y = self.__df_2_components.columns[1]
        idx = self.__df_2_components.columns[2]
        # !!! already changed
        return x, y, idx, self.__df_2_components.copy()