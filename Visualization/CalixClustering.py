"""
First clustering file
"""
import importlib
import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import calix_visual_settings as CVS

importlib.reload(CVS)

def import_csv_and_craft(csv_file_directory,
                         csv_file_name,
                         ordered_target_column_list,
                         ordered_data_column_list,
                         target_mode,
                         data_mode):
    """
    A function that opens a .csv file and populates a pandas DataFrame with the desired
    descriptive ('target') columns along with the desired data columns. For both the target
    and columns, there are 3 modes for describing what to include or exclude.
    In 'exact', the column_list names are used precisely. In 'exclude', the columns included
    are everything except those listed. In 'range', an inclusive range of positions is used
    to avoid creating very long exact lists. In 'all' mode, the starting .csv is returned 
    with no editing performed.

    When using the 'exclude' or 'all' approach, *only* the target_column list is read - as the .csv
    file doesn't know which columns are target vs. data

    Parameters
    ----------
    csv_file_directory : TYPE
        DESCRIPTION.
    csv_file_name : TYPE
        DESCRIPTION.
    ordered_target_column_list : TYPE
        DESCRIPTION.
    ordered_data_column_list : TYPE
        DESCRIPTION.
    target_mode : TYPE
        DESCRIPTION.
    data_mode : TYPE
        DESCRIPTION.

    Returns
    -------
    A pandas DataFrame organized according to the input parameters

    """

    starting_frame = pd.read_csv(csv_file_directory + csv_file_name, header=0, index_col=0)
    
    if target_mode == 'all':
        return starting_frame
    
    populated_frame = pd.DataFrame(index=starting_frame.index)
    
    if target_mode == 'exact':
        for target_title in ordered_target_column_list:
            populated_frame[target_title] = starting_frame[target_title]
    elif target_mode == 'range':
        target_titles = parse_range_list(ordered_target_column_list,
                                         starting_frame)
        for single_title in target_titles:
            populated_frame[single_title] = starting_frame[single_title]
    elif target_mode == 'exclude':
        for possible_title in starting_frame.columns:
            if possible_title not in ordered_target_column_list:
                populated_frame[possible_title] = starting_frame[possible_title]
        return populated_frame
    else:
        raise ValueError('Incorrect target mode used')

    if data_mode == 'exact':
        for target_title in ordered_data_column_list:
            populated_frame[target_title] = starting_frame[target_title]
    elif data_mode == 'range':
        target_titles = parse_range_list(ordered_data_column_list,
                                         starting_frame)
        for single_title in target_titles:
            populated_frame[single_title] = starting_frame[single_title]
    else:
        raise ValueError('Incorrect data mode used')
        
    return populated_frame

def parse_range_list(ordered_list,
                     full_dataframe):
    """
    A function that takes a list of numerical ranges and single entries ([1, 3, 5-8, 11, 17-19, etc.])
    and a dataframe that will be processed with these ranges, and creates and new list of the column
    titles that correspond to these ranges.

    Parameters
    ----------
    ordered_list : TYPE
        DESCRIPTION.
    full_dataframe : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Create a list to hold the column names
    column_names = []
    # Iterate through the ordered list
    for item in ordered_list:
        # Try to convert the item to an integer. This will be successful if it is a single entry
        try:
            curr_int = int(item)
            # If successful, append the column name to the list
            column_names.append(full_dataframe.columns[curr_int])
        # If it is not a single entry, it will be a range
        except ValueError:
            # Split the range into a list of two integers
            start, end = item.split('-')
            # Convert the integers to integers
            start = int(start)
            end = int(end)
            # Append the column names to the list
            column_names.extend(full_dataframe.columns[start:end+1])
    # Return the list of column names
    return column_names

def create_umap_cluster_frame(starting_dataframe,
                              n_neighbors,
                              min_dist):
    """
    

    Parameters
    ----------
    csv_file_directory : TYPE
        DESCRIPTION.
    csv_file_name : TYPE
        DESCRIPTION.
    umap_cluster_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    reducer = umap.UMAP(metric='euclidean',
                        spread=3,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist)
    
    #Remove labels from dataframe
    #data_only_frame = starting_dataframe.drop(columns=[col for col in starting_dataframe.columns if col in umap_cluster_dict['target_column_list']])

    #Create x and y plot values from UMAP
    scatter_data = reducer.fit_transform(starting_dataframe)

    starting_dataframe['X'] = scatter_data[:, 0]
    starting_dataframe['Y'] = scatter_data[:, 1]
                         
    return starting_dataframe

def create_umap_scatter(umap_cluster_dataframe,
                        scatter_plot_dict,
                        output_name):
    """
    

    Parameters
    ----------
    umap_cluster_dataframe : TYPE
        DESCRIPTION.
    scatter_plot_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """                         
    #Extract scatter points                             
    x_val = umap_cluster_dataframe['X']
    y_val = umap_cluster_dataframe['Y']
    
    #Set up scatter plot
    fig, ax = plt.subplots(figsize=(scatter_plot_dict['plot_width'],
                                    scatter_plot_dict['plot_height']))

    # Use the first letter of each index entry to determine the color.
    marker_color_list = [
        sns.color_palette(
            scatter_plot_dict['calix_color'][str(idx)[0]][0],
            scatter_plot_dict['calix_color'][str(idx)[0]][1]
        )[scatter_plot_dict['calix_color'][str(idx)[0]][2]]
        for idx in umap_cluster_dataframe.index
    ]
    opacity = scatter_plot_dict['marker_opacity']
    marker_type = scatter_plot_dict['marker_type']    
    
    for x, y, color in zip(x_val, y_val, marker_color_list):
        plt.scatter(x, y, color=color, alpha=opacity, marker=marker_type)
    
    unique_letters = sorted(umap_cluster_dataframe.index.astype(str).str[0].unique())
    legend_handles = []
    for letter in unique_letters:
        palette_name, num_shades, shade_index = scatter_plot_dict['calix_color'][letter]
        color = sns.color_palette(palette_name, num_shades)[shade_index]
        handle = plt.Line2D([], [], marker=marker_type, color=color, linestyle='None',
                            markersize=10, label=letter, alpha=opacity)
        legend_handles.append(handle)

    # Add the legend to the plot
    ax.legend(handles=legend_handles, title='Calixarene Scaffold')

    # Save the plot with a unique suffix
    plt.savefig(output_name)
    plt.close(fig)

    return                             

def color_by_scaffold_cluster(csv_file_directory,
                              csv_file_name,
                              umap_setting_dict,
                              num_neighbors,
                              min_dist,
                              output_name):
    """
    A function that takes a .csv file with calixarene/peptide binding information, and creates a UMAP scatter plot
    with each calixarene scaffold (i.e. 'A', 'B', 'C', 'D', 'E', 'P') as a different color.

    By default (for publication), we are using all columns for clustering, so no UMAP dictionary is necessary
    """

    starting_frame = import_csv_and_craft(csv_file_directory=csv_file_directory,
                                          csv_file_name=csv_file_name,
                                          ordered_target_column_list=[],
                                          ordered_data_column_list=[],
                                          target_mode='all',
                                          data_mode=None)
    
    umap_filled_frame = create_umap_cluster_frame(starting_dataframe=starting_frame,
                                                  n_neighbors=num_neighbors,
                                                  min_dist=min_dist)
    
    create_umap_scatter(umap_cluster_dataframe=umap_filled_frame,
                        scatter_plot_dict=umap_setting_dict,
                        output_name=output_name)
    
    return

def kmeans_umap_plot(umap_cluster_dataframe,
                     n_clusters,
                     analysis_filename=None):
    """
    Perform k-means clustering on UMAP results and plot the clusters.
    
    Parameters:
    - df : pandas DataFrame
        DataFrame containing 'X' and 'Y' columns from UMAP.
    - n_clusters : int
        Number of clusters for k-means clustering.
        
    Returns:
    - None (plots the scatterplot)
    """
    
    # Extract 'X' and 'Y' for clustering
    data_to_cluster = umap_cluster_dataframe[['X', 'Y']].values
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data_to_cluster)
    
    # Add cluster labels to original DataFrame and save for analysis
    umap_cluster_dataframe['cluster'] = kmeans.labels_
    umap_cluster_dataframe = umap_cluster_dataframe.sort_values(by='cluster')
    
    # Export file for analysis if appropriate
    if analysis_filename is not None:
        umap_cluster_dataframe.to_csv(analysis_filename)

    # Plotting
    plt.figure(figsize=(10,8))
    sns.scatterplot(x='X', y='Y', hue='cluster', data=umap_cluster_dataframe, palette='viridis', s=60, edgecolor=None, alpha=0.7)
    plt.title(f'UMAP Clustering with k={n_clusters}')

    return

