import csv
from keras.callbacks import Callback
from keras.initializers.initializers import RandomNormal, GlorotNormal
from keras.models import Model
from keras.layers import concatenate, Input, Dropout, LeakyReLU, ELU
from keras.layers.core import Dense
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD, RMSprop, Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from sklearn import preprocessing
pd.options.mode.chained_assignment = None


class RealTimeMetrics(Callback):
    def __init__(self, validation_data):
        super(RealTimeMetrics, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_loss, val_accuracy = self.model.evaluate(self.validation_data, verbose=0)
        print(f'Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}')

def file_read(location):
    df = pd.read_hdf(location, 'df')
    return df


def createModel():
    # Define the standalone models
    ArtistName = Input(shape=(64,))
    GenreIds = Input(shape=(64,))
    LyricistAndComposer = Input(shape=(30,))
    MsnoArtistName = Input(shape=(128,))
    MsnoSongId = Input(shape=(128,))
    Rest = Input(shape=(40,))
    SongLength = Input(shape=(1,))

    artistName = Dense(18, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(ArtistName)
    artistName = Dropout(0.1)(artistName)
    artistName = Model(inputs=ArtistName, outputs=artistName)

    genreIds = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(GenreIds)
    genreIds = Dropout(0.1)(genreIds)
    genreIds = Model(inputs=GenreIds, outputs=genreIds)

    lyricistAndComposer = Dense(15, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(LyricistAndComposer)
    lyricistAndComposer = Dropout(0.1)(lyricistAndComposer)
    lyricistAndComposer = Model(inputs=LyricistAndComposer, outputs=lyricistAndComposer)

    msnoArtistName = Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(MsnoArtistName)
    msnoArtistName = Dropout(0.1)(msnoArtistName)
    msnoArtistName = Model(inputs=MsnoArtistName, outputs=msnoArtistName)

    msnoSongId = Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(MsnoSongId)
    msnoSongId = Dropout(0.1)(msnoSongId)
    msnoSongId = Model(inputs=MsnoSongId, outputs=msnoSongId)

    rest = Dense(20, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(Rest)
    rest = Dropout(0.1)(rest)
    rest = Model(inputs=Rest, outputs=rest)

    songLength = Dense(2, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(SongLength)
    songLength = Dropout(0.1)(songLength)
    songLength = Model(inputs=SongLength, outputs=songLength)

    # Merge the models
    combinedRegular = concatenate(
        [artistName.output, genreIds.output, lyricistAndComposer.output, songLength.output])
    modelRegular = Dense(128, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(combinedRegular)
    modelRegular = Dropout(0.5)(modelRegular)
    modelRegular = Model(inputs=[artistName.input, genreIds.input, lyricistAndComposer.input, songLength.input], outputs=modelRegular)

    combinedCategorical = concatenate([msnoArtistName.output, msnoSongId.output, rest.output])

    modelCategorical = Dense(64, "relu", kernel_initializer=RandomNormal(stddev=0.05))(combinedCategorical)
    modelCategorical = Dropout(0.5)(modelCategorical)
    modelCategorical = Model(inputs=[msnoArtistName.input, msnoSongId.input, rest.input],
                  outputs=modelCategorical)

    combined = concatenate([modelRegular.output, modelCategorical.output])

    model1 = Dense(72, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(combined)
    model1 = Dropout(0.5)(model1)
    model1 = Model(
        inputs=[modelRegular.input, modelCategorical.input],
        outputs=model1)

    # model2 = Dense(128, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(model1.output)
    # model2 = Dropout(0.5)(model2)
    # model2 = Model(inputs=model1.input, outputs=model2)
    #
    # model3 = Dense(64, activation="softmax", kernel_initializer=RandomNormal(stddev=0.05))(model2.output)
    # model3 = Dropout(0.5)(model3)
    # model3 = Model(inputs=model2.input, outputs=model3)
    #
    # model4 = Dense(128, activation="softmax", kernel_regularizer=regularizers.l2(0.001), kernel_initializer=RandomNormal(stddev=0.05))(model3.output)
    # model4 = Model(inputs=model3.input, outputs=model4)

    test = Dense(1, activation="sigmoid", kernel_initializer=GlorotNormal(seed=42))(model1.output)
    test = Model(inputs=model1.input, outputs=test)

    initial_learning_rate = 0.0021  # Initial learning rate
    lr_schedule = ExponentialDecay(
        initial_learning_rate,  # Initial learning rate
        decay_steps=20,  # Decay every 100 steps
        decay_rate=0.9  # Reduce by 10% every time
    )

    test.compile(optimizer=RMSprop(learning_rate=lr_schedule),#0.0015 #Adam(learning_rate=0.0005),#RMSprop(learning_rate=0.002),#SGD(learning_rate=0.01, momentum=0.45),#optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),#.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return test


def get_dummies_and_sort_columns(df, classes):
    df = pd.get_dummies(df.dropna()).reindex(df.index)
    df = df.reindex(columns=list(classes), fill_value=0)
    df.columns = df.columns.astype(str)
    return df.reindex(sorted(df.columns), axis=1)


def select_on_songid(table, train):
    song_ids = pd.DataFrame(train['song_id'])
    table['song_id'] = table.index
    song_ids = song_ids.join(table.set_index('song_id'), on='song_id')
    return song_ids.drop(['song_id'], axis = 1)


def select_on_msno(table, func_train, classes):
    table = pd.DataFrame(table)
    func_train = pd.DataFrame(func_train['msno'])
    table['msno'] = table.index
    func_train = func_train.join(table.set_index('msno'), on='msno')
    func_train = func_train.drop(['msno'], axis=1)
    func_train = get_dummies_and_sort_columns(func_train.iloc[:, 0], classes)
    return func_train


def modify_lyricist_and_composer(lyricist_and_composer, lyricist):
    lyricist['lyricist'] = lyricist['lyricist'].str.normalize('NFKC')
    lyricist_index = lyricist['song_id']
    lyricist_strings = lyricist['lyricist'].str.split(pat='|', expand=True)
    lyricist_strings.index = lyricist['song_id']
    lyricist_dict = lyricist_and_composer.T.to_dict('list')
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
    unique_vars = ["_s_" + str(var) for var in range(29,-1,-1)]
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
    return lyricist_strings


def normalize(table):
    for column in table:
        table[column].fillna(value=(table[column].mean()), inplace=True)

    x = table.values
    ind = table.index
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    table = pd.DataFrame(x_scaled)
    table.index = ind
    for column in table:
        table[column].fillna(value=(table[column].mean()), inplace=True)
    return table


def prepare_rest(train, source_system_tab_uniques, source_screen_name_uniques, source_type_uniques):
    source_system_tab = get_dummies_and_sort_columns(train['source_system_tab'], source_system_tab_uniques)
    source_screen_name = get_dummies_and_sort_columns(train['source_screen_name'], source_screen_name_uniques)
    source_type = get_dummies_and_sort_columns(train['source_type'], source_type_uniques)

    final_train = pd.concat([source_system_tab, source_screen_name], axis=1)
    final_train = pd.concat([final_train, source_type], axis=1)
    final_train = final_train.drop(['nan'], axis=1)

    return final_train


def add_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def generate_data(train, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, split):
    while not train.empty:
        partial_train = train[:split]
        train = train[split:]

        artistName = add_noise(normalize(select_on_songid(artist_name, partial_train)))
        genreIds = normalize(select_on_songid(genre_ids, partial_train))
        lyricistAndComposer = add_noise(normalize(select_on_songid(lyricist_and_composer,
                                                         partial_train)))
        songLength = normalize(select_on_songid(song_length, partial_train))
        # categorical
        msnoArtistName = normalize(select_on_msno(msno_artist_name, partial_train, msno_artist_name_classes))
        # categorical
        msnoSongId = normalize(select_on_msno(msno_and_song_id, partial_train, msno_and_song_id_classes))
        rest = normalize(prepare_rest(partial_train, source_system_tab_uniques, source_screen_name_uniques,
                                      source_type_uniques))
        y = partial_train[['target']]
        yield ([artistName.to_numpy(), genreIds.to_numpy(), lyricistAndComposer.to_numpy(),
               songLength.to_numpy(), msnoArtistName.to_numpy(), msnoSongId.to_numpy(), rest.to_numpy()], y.to_numpy())



song_length = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/songs.csv', names=['song_id', 'song_length',
                                                                                           'genre_ids', 'artist_name',
                                                                                           'composer', 'lyricist',
                                          'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE, header=1)

train = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)

artist_name = normalize(file_read('artist_name_64PCA.h5'))

genre_ids = normalize(file_read('genre_ids_64PCA.h5'))

lyricist_and_composer = normalize(file_read('lyricist_and_composer_with_song_id_new.h5'))

song_length = pd.DataFrame(song_length[['song_id','song_length']])
max_length = song_length['song_length'].quantile(.95)
song_length['song_length'] = np.where(song_length['song_length'] > max_length, max_length, song_length['song_length'])
song_length.index = song_length['song_id']

msno_artist_name = file_read('msno_and_artist_name.h5')
msno_artist_name = msno_artist_name.squeeze()
msno_artist_name_classes = msno_artist_name.unique()

msno_and_song_id = file_read('msno_and_song_id.h5')
msno_and_song_id = msno_and_song_id.squeeze()
msno_and_song_id_classes = msno_and_song_id.unique()

source_system_tab_uniques = train['source_system_tab'].unique()
source_screen_name_uniques = train['source_screen_name'].unique()
source_type_uniques = train['source_type'].unique()

#msno song_id source_system_tab source_screen_name source_type target
split = len(train.index)//25
train = train.sample(frac=1)
test = train[:split]
train = train[split:]

split = 16384
#normalize ola eks arxhs
model = createModel()

# Assuming you have a validation generator similar to your training generator
real_time_metrics = RealTimeMetrics(generate_data(test, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, split))

print(model.summary())
model.fit(generate_data(train, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, split), steps_per_epoch=len(train)//split,
                                  epochs=10,
                                  callbacks=[real_time_metrics]
                                  )