import csv
import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, LeakyReLU
import gc
import pandas as pd
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


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

    return main_table.sample(frac=1), un, ind


def createModel(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder layers
    hidden_layer1 = Dense(4096)(input_layer)
    hidden_layer1 = LeakyReLU(alpha=0.01)(hidden_layer1)
    hidden_layer1 = Dropout(0.5)(hidden_layer1)

    hidden_layer2 = Dense(2048)(hidden_layer1)
    hidden_layer2 = LeakyReLU(alpha=0.01)(hidden_layer2)
    hidden_layer2 = Dropout(0.5)(hidden_layer2)

    hidden_layer3 = Dense(1024)(hidden_layer2)
    hidden_layer3 = LeakyReLU(alpha=0.01)(hidden_layer3)
    hidden_layer3 = Dropout(0.5)(hidden_layer3)

    hidden_layer4 = Dense(512)(hidden_layer3)
    hidden_layer4 = LeakyReLU(alpha=0.01)(hidden_layer4)
    hidden_layer4 = Dropout(0.5)(hidden_layer4)
    encoded = Dense(encoding_dim, activation='relu')(hidden_layer4)
    encoded = Dropout(0.5)(encoded)

    # Decoder layers
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    decoded = Dropout(0.5)(decoded)

    custom_optimizer = Adam(lr=0.001)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=custom_optimizer, loss='mse')

    # Encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder


# Generates training data
def generate_data(df, batch, un, num_epochs = 10):
    while True:
        for _ in range(num_epochs):
            shuffled_df = df.sample(frac=1, random_state=42).copy()
            while not shuffled_df.empty:
                df_chunk = shuffled_df[:batch]
                shuffled_df = shuffled_df[batch:]

                df_chunk = (df_chunk.str.split('|', expand=True)
                            .stack()
                            .groupby(level=0)
                            .value_counts()
                            .unstack(fill_value=0))

                df_chunk = df_chunk.apply(lambda x: (x > 0).astype(int))
                df_chunk = pd.DataFrame(df_chunk.T.reindex(un).T.to_numpy()).fillna(0)
                yield df_chunk, df_chunk


# Generates test data
def generate_data_test(df, batch, un):
    while True:
        shuffled_df = df.sample(frac=1, random_state=42).copy()
        while not shuffled_df.empty:
            df_chunk = shuffled_df[:batch]
            shuffled_df = shuffled_df[batch:]

            df_chunk = (df_chunk.str.split('|', expand=True)
                        .stack()
                        .groupby(level=0)
                        .value_counts()
                        .unstack(fill_value=0))

            df_chunk = df_chunk.apply(lambda x: (x > 0).astype(int))
            df_chunk = pd.DataFrame(df_chunk.T.reindex(un).T.to_numpy()).fillna(0)
            yield df_chunk, df_chunk


# Main
myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
train, un, ind = prepare_table()
length = len(train)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder, encoder = createModel(input_dim=len(un), encoding_dim=256)

validation_split = 0.15
train_copy = train
train = train.sample(frac=1, random_state=42).copy()


num_validation = int(validation_split * len(train))
test = train[:num_validation]
train = train[num_validation:]

batch_size = 128
print(autoencoder.summary())
train_generator = generate_data(train, batch_size, un, 10)
validation_generator = generate_data_test(test, batch_size, un)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
autoencoder.fit(train_generator, steps_per_epoch=10, epochs=10,
                validation_data=validation_generator, validation_steps=1, callbacks=[early_stopping])

train = train_copy
del train_copy, test
finalArray = pd.DataFrame()

while not train.empty:
    partial_array = train[:batch_size]
    train = train[batch_size:]
    partial_array = (partial_array.str.split('|', expand=True)
                     .stack()
                     .groupby(level=0)
                     .value_counts()
                     .unstack(fill_value=0))
    partial_array = partial_array.apply(lambda x: (x > 0).astype(int))
    partial_array = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy()).fillna(0)
    finalArray = pd.concat([finalArray, pd.DataFrame(encoder.predict(partial_array))])

scaler = MinMaxScaler()
finalArray = scaler.fit_transform(finalArray)  # Normalize using Min-Max scaler
finalArray = pd.DataFrame(KMeans(n_clusters=128, random_state=0, n_init="auto").fit_predict(finalArray))
finalArray.index = ind

finalArray.to_hdf(myPath + 'msno_artist_name_autoencoder_1.h5', key='df', mode='w')

# 4 tables have been produced
#1 ---> one hidden layer of 4096, final dim of 256 similarity to kmeans (no pca) of 6.8% (msno_artist_name_autoencoder_1)
#2 ---> four hidden layers of 4096, 2048, 1024, 512, final dim of 256 similarity to kmeans (no pca) of 6.8% (msno_artist_name_autoencoder_2)
#3 --->no hidden layers final dim of 256, similarity to kmeans (no pca) of 7% (msno_artist_name_autoencoder_3)
#4 ---> no hidden layers, final dim of 512, similarity to kmeans (no pca) of 7% (msno_artist_name_autoencoder_4)

# Similarity to results produced by kmeans having ran PCA first were near 11%