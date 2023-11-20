import csv
import gc
import pandas as pd
from sklearn.cluster import KMeans


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
    songs = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/songs.csv', names=['song_id', 'song_length',
                                                                                               'genre_ids', 'artist_name',
                                                                                               'composer', 'lyricist',
                                              'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE, header=1)
    main_table = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)
    main_table = pd.merge(main_table, songs, on='song_id', how='left')
    del songs
    main_table = main_table[['msno', 'artist_name']]
    main_table['artist_name'] = main_table['artist_name'].str.normalize('NFKC')
    main_table = main_table.assign(artist_name=main_table['artist_name'].str.split('|')).explode('artist_name')
    gc.collect()
    main_table = main_table.fillna('UNKNOWN')
    artist_counts = main_table['artist_name'].value_counts()
    total_users = len(main_table['msno'].unique())
    threshold = 0.001 * total_users
    unpopular_artists = artist_counts[artist_counts < threshold]
    replace_map = {k: 'OTHER' for k in unpopular_artists.index}
    main_table['artist_name'] = main_table['artist_name'].map(replace_map).fillna(main_table['artist_name'])
    un = pd.unique(main_table['artist_name'].values.ravel('K'))
    main_table = main_table.groupby('msno')['artist_name'].agg('|'.join)
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


train, un, ind = prepare_table()
batch_size = 4096
finalArray = pd.Dataframe()

while not train.empty:
    partial_array = train[:batch_size]
    train = train[batch_size:]
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

finalArray.to_hdf('msno_and_artist_name_kmeans.h5', key='df', mode='w')