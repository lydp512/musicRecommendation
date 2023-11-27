import csv
import gc

import pandas as pd
from sklearn.cluster import KMeans

myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
songs = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/songs.csv', names=['song_id', 'song_length',
                                                                                           'genre_ids', 'artist_name',
                                                                                           'composer', 'lyricist',
                                          'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE, header=1)

songs = songs[['lyricist','composer']]
songs = songs.fillna('UNKNOWN')

#normalizing the strings
lyricist_array = songs['lyricist'].str.normalize('NFKC')
songs['composer'] = songs['composer'].str.normalize('NFKC')

#counting unique values for those that appear more than 11 times
lyricist_array = lyricist_array.str.split(pat='|', expand=True)
value_counts = lyricist_array.stack().value_counts()
value_counts = value_counts[value_counts < 11]

#if rare value, replace with 'other'
replace_map = {k:'OTHER' for k in value_counts.index}
first_time = True
finalDf = pd.DataFrame()
#map composers and lyricists in a 2d array
for column_name in lyricist_array:
    lyricist_array[column_name] = lyricist_array[column_name].map(replace_map).fillna(lyricist_array[column_name])
    if first_time:
        first_time = False
        composer_with_lyricist = pd.concat([songs['composer'], lyricist_array[column_name]], axis=1)
        composer_with_lyricist = composer_with_lyricist.groupby(column_name)['composer'].agg('|'.join)
        finalDf = composer_with_lyricist.to_frame()
    else:
        composer_with_lyricist = pd.concat([songs['composer'], lyricist_array[column_name]], axis=1)
        composer_with_lyricist = composer_with_lyricist.groupby(column_name)['composer'].agg('|'.join)
        finalDf = finalDf.join(composer_with_lyricist, rsuffix=column_name)

#first two columns are dropped because they are columns of an empty string
un = pd.unique(lyricist_array.values.ravel('K'))
finalDf = finalDf.iloc[2:]
finalDf = finalDf[finalDf.columns].apply(
    lambda x: '|'.join(x.dropna().astype(str)),
    axis=1
)

del composer_with_lyricist
lyricist = pd.DataFrame(songs[['song_id','lyricist']])
del songs
del lyricist_array
gc.collect()

#transform the map into a 2d array of categorical values
split = 70
indexes = list(finalDf.index.values)
finalArray = pd.DataFrame()
while not finalDf.empty:
    partial_array = finalDf[:split]
    finalDf = finalDf[split:]
    partial_array = (partial_array.str.split('|', expand=True)
                             .stack()
                             .groupby(level=0)
                             .value_counts()
                             .unstack(fill_value=0))
    partial_array = pd.DataFrame(partial_array.T.reindex(un).T.fillna(0).to_numpy()) # (34619,1681)
    finalArray = pd.concat([finalArray, partial_array])

#transform the lyricists according to their cluster distance to the composers
finalArray = pd.DataFrame(KMeans(n_clusters=30, random_state=0, n_init="auto").fit_transform(finalArray))
finalArray.index = indexes
# saving it just in case there are memory errors later
finalArray.to_hdf('lyricist_and_composer.h5', key='df', mode='w')

# transform the 2d composer/lyricist matrix in relation to song_id
# if there is only one lyricist per row, then the simple lyricist/composer distance is kept
# if there are more than one, then the average between the lyricists is calculated
# if there are missing values, the total average is used
lyricist['lyricist'] = lyricist['lyricist'].str.normalize('NFKC')
lyricist_index = lyricist['song_id']
lyricist_strings = lyricist['lyricist'].str.split(pat='|', expand=True)
lyricist_strings.index = lyricist['song_id']
lyricist_dict = finalArray.T.to_dict('list')
for col in lyricist_strings.columns:
    lyricist_strings[col] = lyricist_strings[col].map(lyricist_dict)
lyricist_strings.index = lyricist_index

columns = lyricist_strings.columns.tolist()
del columns[0]
lyricist_strings_multiple_cols = lyricist_strings[
    lyricist_strings.loc[:, lyricist_strings.columns != lyricist_strings.columns[0]].notnull().any(1)]
lyricist_strings_multiple_cols_index = lyricist_strings_multiple_cols.index
lyricist_strings = lyricist_strings[
    lyricist_strings.loc[:, lyricist_strings.columns != lyricist_strings.columns[0]].isnull().all(1)]

lyricist_strings = lyricist_strings.iloc[:, 0]
lyricist_strings = pd.DataFrame(dict(zip(lyricist_strings.index, lyricist_strings.values))).T

lyricist_strings_multiple_cols = pd.concat(
    [lyricist_strings_multiple_cols[c].apply(pd.Series).add_prefix(str(c) + "_s_") for c in
     lyricist_strings_multiple_cols], axis=1
)
unique_vars = ["_s_" + str(var) for var in range(len(lyricist_strings_multiple_cols) - 1,-1,-1)]
for var in unique_vars:
    lyricist_strings_multiple_cols[var + '_mean'] = lyricist_strings_multiple_cols.filter(like=var).mean(axis=1)
    lyricist_strings_multiple_cols = lyricist_strings_multiple_cols[
        lyricist_strings_multiple_cols.columns.drop([val for val in list(lyricist_strings_multiple_cols.filter(regex=var)) if
                                      not val.endswith('_mean')])]
lyricist_strings_multiple_cols.index = lyricist_strings_multiple_cols_index

lyricist_strings_multiple_cols = lyricist_strings_multiple_cols.reindex(sorted(lyricist_strings_multiple_cols.columns), axis=1)
lyricist_strings.columns = lyricist_strings.columns.astype(str)
lyricist_strings = lyricist_strings.add_prefix('_s_')
lyricist_strings = lyricist_strings.add_suffix('_mean')
lyricist_strings = lyricist_strings.reindex(sorted(lyricist_strings.columns), axis=1)

lyricist_strings = pd.concat([lyricist_strings, lyricist_strings_multiple_cols])

lyricist_strings.to_hdf('lyricist_and_composer_with_song_id_new.h5', key='df', mode='w')