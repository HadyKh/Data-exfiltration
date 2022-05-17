import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# visualize the data
def visualize_dataset(data):
    """ Visualize the whole dataset
    Parameters:
        data: dataframe
    """
    data = data.drop(['timestamp', 'sld', 'longest_word'], axis = 1)
    # shuffle data
    data_shuffle = data.sample(frac=1).reset_index(drop=True)
    # take a sample of the data
    data_shuffle = data_shuffle.iloc[:1000]

    # visualize data
    tsne = TSNE(n_components=2, init='random', random_state = 0) # applying TSNE to obtain a 2d data
    data_plot = tsne.fit_transform(data_shuffle)
    plt.plot(data_plot)
    plt.savefig('../../reports/figures/01) exploring_data_visualization.jpg') # saving

    # check the balance and unbalanced
def check_balance(data):
    """ Visualize the data to check the balance
    Parameters:
        data: dataframe
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.bar(['1', '0'], data['Label'].value_counts()) # getting the number of the labels
    plt.title('Labels count', fontsize = 'x-large')
    plt.ylabel('Count', fontsize = 'x-large')
    plt.xlabel('Labels', fontsize = 'x-large')
    plt.savefig('../../reports/figures/02) exploring_balance_check.jpg') # saving
 
    # visualize correlation map
def correlation_map(data):
    """ Visualize the correlation map
    Parameters:
        data: dataframe
    """
    plt.figure(figsize=(8,8))
    fig = sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm') # visualizing the correlation in the heatmap
    fig = fig.get_figure()
    fig.savefig('../../reports/figures/03) exploring_correlation_map.jpg') # saving

def implementing_shap(data, model):
    """ Visualizing shap values to get the feature importance
    Parameters:
        data: dataframe
    """
    import shap
    from shap import KernelExplainer
    from shap import Explanation

    data = data.drop(['timestamp', 'sld', 'longest_word'], axis = 1)

    masker = shap.maskers.Independent(data, 100)  # masks out the tabular features
    ke = KernelExplainer(model.predict_proba, data = masker.data) # computing the importance of each feature
    ke.expected_value 
    shap_values = ke.shap_values(masker.data) # getting the values
    shap.summary_plot(shap_values, data.columns, show = False) # visualize
    plt.savefig('../../reports/figures/04) shap_feature_importance.jpg') # saving
