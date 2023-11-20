import pandas as pd


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
auto2 = file_read(myPath + 'msno_artist_name_PCA_KMEANS.h5')
auto1 = file_read(myPath + 'msno_artist_name_autoencoder_2.h5')

# Rename the columns to give them names
auto1.columns = ['cluster_auto1']
auto2.columns = ['cluster_auto2']

# Merge auto1 and auto2 on userID
merged_df = pd.merge(auto1, auto2, left_index=True, right_index=True, how='outer')

# Determine the most common mapping between clusters in auto1 and auto2
common_mapping = (
    merged_df.groupby('cluster_auto1')['cluster_auto2']
    .apply(lambda x: x.mode().iat[0])
    .to_dict()
)

# Map cluster labels in auto1 to the most common labels in auto2
auto1_mapped = auto1['cluster_auto1'].map(common_mapping)

# Check if the clusters are the same for each user
auto1['clusters_match'] = auto1_mapped == auto2['cluster_auto2']
# Calculate the percentage of common values
percentage_common = auto1['clusters_match'].sum() / len(auto1) * 100
# Display the result
# Identify the most successful clusters
most_successful_clusters = (
    auto1[auto1['clusters_match']]
    .groupby('cluster_auto1')
    .size()
    .idxmax()
)

# Display the results
print(f"Percentage of common values: {percentage_common:.2f}%")
print(f"Most successful cluster: {most_successful_clusters}")