import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import csv
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

PositiveSampleFeature = []
ReadMyCsv(PositiveSampleFeature, "PDCSLCS(Dataset4).csv")
NegativeSampleFeature = []
ReadMyCsv(NegativeSampleFeature, "NDCSLCS(Dataset4).csv")

SampleFeature = []
SampleFeature.extend(PositiveSampleFeature)
SampleFeature.extend(NegativeSampleFeature)

SampleFeature = np.array(SampleFeature)
x = SampleFeature
print(len(x))

from sklearn.model_selection import train_test_split
x_train, x_test = train_test_split(x, test_size=0.2)
x_train = x_train.astype('float64') / 1.
x_test = x_test.astype('float64') / 1.

encoding_dim = 64
input_data = Input(shape=(len(SampleFeature[0]),))
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_data)
decoded = Dense(293, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_data, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_data, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

autoencoder.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=300, batch_size=128, shuffle=True, validation_data=(x_test, x_test),callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)])

encoded_data_pre = encoder.predict(x)
storFile(encoded_data_pre, 'DCSLCS(Dataset4)(64dim).csv')

# Similarly, get 'DGSLGS(Dataset4)(64dim).csv' and 'DSSLFS(Dataset4)(64dim).csv'







