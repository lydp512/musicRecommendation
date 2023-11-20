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
    main_table = pd.read_csv('/home/lydia/PycharmProjects/untitled/original data/train.csv', delimiter=',', header=0)
    main_table = main_table[['msno', 'song_id']]
    gc.collect()
    main_table = main_table.fillna('UNKNOWN')
    song_counts = main_table['song_id'].value_counts()
    total_users = len(main_table['msno'].unique())
    threshold = 0.001 * total_users
    unpopular_songIDs = song_counts[song_counts < threshold]
    replace_map = {k: 'OTHER' for k in unpopular_songIDs.index}
    main_table['song_id'] = main_table['song_id'].map(replace_map).fillna(main_table['song_id'])
    un = pd.unique(main_table['song_id'].values.ravel('K'))

    main_table = main_table.groupby('msno')['song_id'].agg('|'.join)
    ind = main_table.index
    return main_table, un, ind


def createModel(input_dim, encoding_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # # Encoder layers
    hidden_layer1 = Dense(4096)(input_layer)
    hidden_layer1 = LeakyReLU(alpha=0.01)(hidden_layer1)  # Adjust alpha as needed
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
    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=custom_optimizer, loss='mse')

    # Encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder


# Function to generate data for training the autoencoder
def generate_data(df, batch, un, num_epochs = 10):
    while True:  # Loop indefinitely
        for _ in range(num_epochs):
            shuffled_df = df.sample(frac=1, random_state=42).copy()  # Shuffle the dataframe
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


def generate_data_test(df, batch, un, num_epochs = 10):
    while True:  # Loop indefinitely for validation
        shuffled_df = df.sample(frac=1, random_state=42).copy()  # Shuffle the dataframe
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


# Main part of the code
myPath = '/home/lydia/PycharmProjects/musicRecommendationFiles/'
train, un, ind = prepare_table()
length = len(train)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder, encoder = createModel(input_dim=len(un), encoding_dim=256)

validation_split = 0.15
train = train.sample(frac=1, random_state=42).copy()
og_train = train
num_validation = int(validation_split * len(train))
test = train[:num_validation]
train = train[num_validation:]

batch_size = 128
print(autoencoder.summary())
train_generator = generate_data(train, batch_size, un, 10)
validation_generator = generate_data_test(test, batch_size, un, 10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)
autoencoder.fit(train_generator, steps_per_epoch=10, epochs=10,
                validation_data=validation_generator, validation_steps=1, callbacks=[early_stopping])

i = 0
first_time = True
print("yuh yuh yuh")
train = og_train
del og_train
batch_size = 2048
while not train.empty:
    print(((i + 1) * batch_size) / length)
    partial_array = train[:batch_size]
    train = train[batch_size:]
    if first_time:
        partial_array = (partial_array.str.split('|', expand=True)
                         .stack()
                         .groupby(level=0)
                         .value_counts()
                         .unstack(fill_value=0))

        partial_array = partial_array.apply(lambda x: (x > 0).astype(int))
        finalArray = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy()).fillna(0)
        finalArray = pd.DataFrame(encoder.predict(finalArray))
        first_time = False
    else:
        partial_array = (partial_array.str.split('|', expand=True)
                         .stack()
                         .groupby(level=0)
                         .value_counts()
                         .unstack(fill_value=0))
        partial_array = partial_array.apply(lambda x: (x > 0).astype(int))
        partial_array = pd.DataFrame(partial_array.T.reindex(un).T.to_numpy()).fillna(0)
        finalArray = pd.concat([finalArray, pd.DataFrame(encoder.predict(partial_array))])
    i = i + 1

scaler = MinMaxScaler()
finalArray = scaler.fit_transform(finalArray)  # Normalize using Min-Max scaler
finalArray = pd.DataFrame(KMeans(n_clusters=128, random_state=0, n_init="auto").fit_predict(finalArray))
finalArray.index = ind

finalArray.to_hdf(myPath + 'msno_song_id_autoencoder_4.h5', key='df', mode='w')

#1 no hidden layers, 512 output dim
#3 no hidden layers, 256 output dim
#2 one hidden layer pf 4096, 256 output dim 0.077 loss
# 4 all