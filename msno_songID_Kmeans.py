import gc
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, IncrementalPCA


def get_dummies_and_sort_columns(df, classes):
    df = pd.get_dummies(df.dropna(), prefix=None).reindex(df.index)
    df.columns = df.columns.str.replace('song_id_', '')
    df = df.reindex(columns=list(classes), fill_value=0)
    df.columns = df.columns.astype(str)
    df = df.astype(bool)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.fillna(False)
    return df


def prepare_table():
    main_table = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)
    main_table = main_table[['msno', 'song_id']]
    gc.collect()
    main_table = main_table.fillna('UNKNOWN')
    song_counts = main_table['song_id'].value_counts()
    total_users = len(main_table['msno'].unique())
    threshold = 0.002 * total_users
    unpopular_songIDs = song_counts[song_counts < threshold]
    replace_map = {k: 'OTHER' for k in unpopular_songIDs.index}
    main_table['song_id'] = main_table['song_id'].map(replace_map).fillna(main_table['song_id'])
    un = pd.unique(main_table['song_id'].values.ravel('K'))

    main_table = main_table.groupby('msno')['song_id'].agg('|'.join)
    ind = main_table.index
    return main_table, un, ind


def generate_batch(main, batchSize, ln):
    i = 0
    while not main.empty:
        print(((i + 1) * batchSize) / ln)
        batch = main[:batchSize]
        main = main[batchSize:]
        yield batch


def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
train, un, ind = prepare_table()
indexes = list(train.index.values)
finalArray = pd.DataFrame()
chunk_size = 4096
while not train.empty:
    partial_array = train[:chunk_size]
    train = train[chunk_size:]
    partial_array = (partial_array.str.split('|', expand=True)
                     .stack()
                     .groupby(level=0)
                     .value_counts()
                     .unstack(fill_value=0))

    partial_array = partial_array.apply(lambda x: (x > 0).astype(int))
    partial_array = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy())
    finalArray = pd.concat([finalArray.astype(bool), partial_array.astype(bool)])

finalArray = pd.DataFrame(KMeans(n_clusters=128, random_state=0, n_init="auto").fit_predict(finalArray))
finalArray.index = ind

finalArray.to_hdf(myPath + 'msno_and_song_id_kmeans_v2.h5', key='df', mode='w')
