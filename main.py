import csv
from keras.callbacks import Callback, EarlyStopping
from keras.initializers.initializers import RandomNormal, GlorotNormal
from keras.models import Model
from keras.layers import concatenate, Input, Dropout, LeakyReLU, ELU
from keras.layers.core import Dense
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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
    #still testing this
    # Define input layers
    ArtistName = Input(shape=(64,))
    GenreIds = Input(shape=(64,))
    LyricistAndComposer = Input(shape=(30,))
    SongLength = Input(shape=(1,))
    Rest = Input(shape=(40,))
    MsnoArtistName = Input(shape=(128,))
    MsnoSongId = Input(shape=(128,))

    shared_layers = []

    artistName = Dense(128, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(ArtistName)
    artistName = Dropout(0.3)(artistName)
    shared_layers.append(Model(inputs=ArtistName, outputs=artistName))

    genreIds = Dense(128, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(GenreIds)
    genreIds = Dropout(0.3)(genreIds)
    shared_layers.append(Model(inputs=GenreIds, outputs=genreIds))

    lyricistAndComposer = Dense(64, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(
        LyricistAndComposer)
    lyricistAndComposer = Dropout(0.3)(lyricistAndComposer)
    shared_layers.append(Model(inputs=LyricistAndComposer, outputs=lyricistAndComposer))

    songLength = Dense(2, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(SongLength)
    songLength = Dropout(0.3)(songLength)
    shared_layers.append(Model(inputs=SongLength, outputs=songLength))

    rest = Dense(72, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(Rest)
    rest = Dropout(0.3)(rest)
    shared_layers.append(Model(inputs=Rest, outputs=rest))

    msnoArtistName = Dense(256, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(
        MsnoArtistName)
    msnoArtistName = Dropout(0.3)(msnoArtistName)
    shared_layers.append(Model(inputs=MsnoArtistName, outputs=msnoArtistName))

    msnoSongId = Dense(256, activation=LeakyReLU(alpha=0.01), kernel_initializer=RandomNormal(stddev=0.05))(MsnoSongId)
    msnoSongId = Dropout(0.3)(msnoSongId)
    shared_layers.append(Model(inputs=MsnoSongId, outputs=msnoSongId))

    # Merge the models
    combinedRegular = concatenate([layer.output for layer in shared_layers])
    modelRegular = Dense(512, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(combinedRegular)
    modelRegular = Dropout(0.35)(modelRegular)
    modelRegular = Model(inputs=[layer.input for layer in shared_layers], outputs=modelRegular)

    # combinedRegular = concatenate([layer.output for layer in shared_layers])
    # modelRegular = Dense(64, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(combinedRegular)
    # modelRegular = Dropout(0.45)(modelRegular)
    # modelRegular = Model(inputs=[layer.input for layer in shared_layers], outputs=modelRegular)
    #
    # combinedMsno = concatenate([msnoArtistName.output, msnoSongId.output])
    #
    # modelCategorical = Dense(64, "relu", kernel_initializer=RandomNormal(stddev=0.05))(combinedMsno)
    # modelCategorical = Dropout(0.45)(modelCategorical)
    # modelCategorical = Model(inputs=[msnoArtistName.input, msnoSongId.input], outputs=modelCategorical)
    #
    # combined = concatenate([modelRegular.output, modelCategorical.output])

    # model1 = Dense(64, activation="relu", kernel_initializer=RandomNormal(stddev=0.05))(modelRegular)
    # model1 = Dropout(0.15)(model1)
    # model1 = Model(inputs=[modelRegular.input], outputs=model1)

    test = Dense(1, activation="sigmoid", kernel_initializer=GlorotNormal(seed=42))(modelRegular.output)
    test = Model(inputs=modelRegular.input, outputs=test)

    initial_learning_rate = 0.00035
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=432,
        decay_rate=0.9
    )

    test.compile(optimizer=Adam(learning_rate=lr_schedule),
                 loss="binary_crossentropy",
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


def prepare_rest(df, source_system_tab_uniques, source_screen_name_uniques, source_type_uniques):
    source_system_tab = get_dummies_and_sort_columns(df['source_system_tab'], source_system_tab_uniques)
    source_screen_name = get_dummies_and_sort_columns(df['source_screen_name'], source_screen_name_uniques)
    source_type = get_dummies_and_sort_columns(df['source_type'], source_type_uniques)

    final_train = pd.concat([source_system_tab, source_screen_name], axis=1)
    final_train = pd.concat([final_train, source_type], axis=1)
    final_train = final_train.drop(['nan'], axis=1)

    return final_train


def add_noise(data, noise_level=0.02):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def generate_data(df, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, split):
    while True:
        shuffled_df = df.sample(frac=1, random_state=42).copy()
        while not shuffled_df.empty:
            partial_train = shuffled_df[:split]
            shuffled_df = shuffled_df[split:]

            artistName = normalize(add_noise(select_on_songid(artist_name, partial_train)))
            genreIds = normalize(select_on_songid(genre_ids, partial_train))
            lyricistAndComposer = normalize(add_noise(select_on_songid(lyricist_and_composer,
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
                    songLength.to_numpy(), rest.to_numpy(), msnoArtistName.to_numpy(), msnoSongId.to_numpy()],
                   y.to_numpy())


def generate_data_test(df, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, split):
    while True:
        shuffled_df = df.sample(frac=1, random_state=42).copy()
        while not shuffled_df.empty:
            partial_train = shuffled_df[:split]
            shuffled_df = shuffled_df[split:]

            artistName = normalize(add_noise(select_on_songid(artist_name, partial_train)))
            genreIds = normalize(select_on_songid(genre_ids, partial_train))
            lyricistAndComposer = normalize(add_noise(select_on_songid(lyricist_and_composer,
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
                    songLength.to_numpy(), rest.to_numpy(), msnoArtistName.to_numpy(), msnoSongId.to_numpy()],
                   y.to_numpy())


model = createModel()
myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
song_length = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/songs.csv', names=['song_id', 'song_length',
                                                                                           'genre_ids', 'artist_name',
                                                                                           'composer', 'lyricist',
                                          'language'], sep=',|\n', engine='python', quoting=csv.QUOTE_NONE, header=1)

train = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)

artist_name = normalize(file_read(myPath + 'artist_name_64PCA.h5')) #ignore this in later tests

genre_ids = normalize(file_read(myPath + 'genre_ids_64PCA.h5'))

lyricist_and_composer = normalize(file_read(myPath + 'lyricist_and_composer_with_song_id_new.h5'))

song_length = pd.DataFrame(song_length[['song_id','song_length']])
max_length = song_length['song_length'].quantile(.95)
song_length['song_length'] = np.where(song_length['song_length'] > max_length, max_length, song_length['song_length'])
song_length.index = song_length['song_id']

msno_artist_name = file_read(myPath + 'msno_artist_name_autoencoder_2.h5')
msno_artist_name = msno_artist_name.squeeze()
msno_artist_name_classes = msno_artist_name.unique()

msno_and_song_id = file_read(myPath + 'msno_song_id_autoencoder_4.h5')
msno_and_song_id = msno_and_song_id.squeeze()
msno_and_song_id_classes = msno_and_song_id.unique()

source_system_tab_uniques = train['source_system_tab'].unique()
source_screen_name_uniques = train['source_screen_name'].unique()
source_type_uniques = train['source_type'].unique()

#only reducing the train for test purposes
split = len(train.index)//50
train = train.sample(frac=1)
train = train[:split]

unique_msno_values = train['msno'].unique()

# Split train and test
train_msno, test_msno = train_test_split(unique_msno_values, test_size=0.25, random_state=42)
test = train[~train['msno'].isin(train_msno)]
train = train[train['msno'].isin(train_msno)]

batch_size = 1024

# Callbacks
test_data_generator = generate_data_test(test, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                   msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, batch_size)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3,
                               restore_best_weights=True)
print(model.summary())
model.fit(generate_data(train, artist_name, genre_ids, lyricist_and_composer, song_length, msno_artist_name,
                                  msno_and_song_id, source_system_tab_uniques, source_screen_name_uniques,
                                  source_type_uniques, batch_size), steps_per_epoch=len(train)//batch_size,
                                  epochs=10, validation_data=test_data_generator, validation_steps=40,
                                  callbacks=[early_stopping])
