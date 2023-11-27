import csv
import gc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

songs = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/songs.csv', names=['song_id', 'song_length',
                                                                                           'genre_ids', 'artist_name',
                                                                                           'composer', 'lyricist',
                                          'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE, header=1)
train = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)

songs = songs[['song_id', 'artist_name']]
song_map = dict(zip(songs['song_id'], songs['artist_name']))
train['song_id'] = train['song_id'].map(song_map)
train = train.rename(columns={'song_id': 'artist_name'})
train = train[['msno', 'song_id', 'target']]

train = train.fillna('UNKNOWN')

value_counts = train['song_id'].value_counts()
value_counts = value_counts[value_counts < 75]
replace_map = {k: 'OTHER' for k in value_counts.index}
train['song_id'] = train['song_id'].map(replace_map).fillna(train['song_id'])
un = pd.unique(train['song_id'].values.ravel('K'))


split = 50000
first_time = True

# indexes = list(train.index.values)
finalArrayPositive = pd.DataFrame()
finalArrayNegative = pd.DataFrame()

positive_target = train.loc[train['target'] == 1]
negative_target = train.loc[train['target'] == 0]
del train

positive_target = positive_target.groupby('msno')['song_id'].agg('|'.join)
negative_target = negative_target.groupby('msno')['song_id'].agg('|'.join)

positive_indexes = list(positive_target.index.values)
negative_indexes = list(negative_target.index.values)

del positive_target
del negative_target
del un
del first_time
del split
del value_counts
del replace_map

first_time = True
i = 0
while not negative_target.empty:
    partial_array = negative_target[:split]
    negative_target = negative_target[split:]
    partial_array = (partial_array.str.split('|', expand=True)
                     .stack()
                     .groupby(level=0)
                     .value_counts()
                     .unstack(fill_value=0))

    finalArrayNegative = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy())
    i = i + 1


finalArrayNegative.to_hdf('finalArrayNegative75.h5',
                     key='df', mode='w')

first_time = True
# finalArrayPositive = pd.HDFStore('finalArrayPositive.h5')
i = 0
while not positive_target.empty:
    partial_array = positive_target[:split]
    positive_target = positive_target[split:]

    partial_array = (partial_array.str.split('|', expand=True)
                     .stack()
                     .groupby(level=0)
                     .value_counts()
                     .unstack(fill_value=0))
    if first_time:
        finalArrayPositive = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy())
        first_time = False
    else:
        partial_array = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy())
        finalArrayPositive = pd.concat([finalArrayPositive, partial_array])
finalArrayPositive.to_hdf('finalArrayPositive75.h5',
                     key='df', mode='w')

finalArrayPositive = pd.read_hdf('finalArrayPositive.h5')
finalArrayPositive.index = positive_indexes

finalArrayNegative = pd.read_hdf('finalArrayNegative.h5')
finalArrayNegative.index = negative_indexes

finalArrayNegative = finalArrayNegative.mul(-1)

split = 1000
first_time = True
while not finalArrayNegative.empty:
    partial_array = finalArrayNegative[:split]
    finalArrayNegative = finalArrayNegative[split:]

    testlist = list(partial_array.index)

    partial_positive = finalArrayPositive[finalArrayPositive.index.isin(testlist)]
    finalArrayPositive = finalArrayPositive.loc[~finalArrayPositive.index.isin(testlist)]
    partial_array = pd.concat([partial_array, partial_positive])
    partial_array['index1'] = partial_array.index

    if first_time:
        finalArray = partial_array.groupby(['index1']).sum()
        first_time = False
    else:
        finalArray = pd.concat([finalArray, partial_array.groupby(['index1']).sum()])

    print("ok" + str(len(finalArrayNegative)))

if not finalArrayPositive.empty:
    finalArrayPositive['index1'] = finalArrayPositive.index
    finalArray = pd.concat([finalArray, finalArrayPositive.groupby(['index1']).sum()])

del finalArrayNegative
del finalArrayPositive
gc.collect()

indexes = list(finalArray.index.values)
print("starting pca")
finalArray = PCA(n_components=1024).fit_transform(finalArray)

print("starting kmeans")
finalArray = pd.DataFrame(KMeans(n_clusters=128, random_state=0, n_init="auto").fit_predict(finalArray))
finalArray.index = indexes
finalArray.to_hdf('msno_and_song_id.h5', key='df', mode='w')
