from keras import backend as K
import tensorflow as tf
from keras import initializers,layers,regularizers
from keras.layers import Dropout
from keras import callbacks
from keras.models import *
from keras.engine.topology import Layer
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split
import keras
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy import interp
from keras.layers import Dense,Input,LSTM,Bidirectional,Conv1D,BatchNormalization
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale
from keras.layers import Lambda
import math




def MyConfusionMatrix(y_real,y_predict):
    CM = confusion_matrix(y_real, y_predict)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    Acc = (TN + TP) / (TN + TP + FN + FP)
    F1 = (2*(TP / (TP + FP))*(TP / (TP + FN)))/((TP / (TP + FP))+(TP / (TP + FN)))
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(F1, 4))
    Result.append(round(MCC, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumF1 = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumF1 = SumF1 + matrix[counter][1]
        SumMcc = SumMcc + matrix[counter][2]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageF1:', SumF1 / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return


SampleFeature_original=np.loadtxt('Feature(Dataset1).txt',dtype=np.float64)
SampleFeature=np.array(SampleFeature_original)
SampleFeature=minmax_scale(SampleFeature)

SampleLabel_original=np.loadtxt('Label(Dataset1).txt',dtype=np.int)
sample_label=[]
for i in range(len(SampleLabel_original)):
    if SampleLabel_original[i]==1:
        sample_label.append([0,1])
    else:
        sample_label.append([1,0])
sample_label=np.asarray(sample_label)

sample_data_old=SampleFeature.reshape(9034,3,64)
sample_data=sample_data_old.transpose((0,2,1))
sample_data=np.expand_dims(sample_data,axis=2)



class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W')
        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))
            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv1D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,name='PrimaryCaps_conv1d')(inputs)
    output=BatchNormalization()(output)
    output=Dropout(0.2)(output)
    outputs = layers.Reshape(target_shape=[-1, dim_vector], name='PrimaryCaps_reshape')(output)
    return layers.Lambda(squash, name='PrimaryCaps_squash')(outputs)


def new(input):
    return tf.expand_dims(input,-1)


def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    temp = layers.GlobalAveragePooling2D()(x)
    temp = layers.Dense(int(x.shape[-1]) * 5, use_bias=False,activation=keras.activations.relu)(temp)
    temp = layers.Dense(int(x.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(temp)
    att = layers.Multiply()([temp,x])
    after_att = layers.Reshape(target_shape=[64,3], name='after_attention')(att)
    conv1 = layers.Conv1D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu', name='Conv1D')(after_att)
    conv1=BatchNormalization()(conv1)
    conv1=Dropout(0.2)(conv1)
    lstm = layers.Bidirectional(LSTM(units=64,   name='BiLSTM',  return_sequences=True))(conv1)
    primarycaps = PrimaryCap(lstm, dim_vector=8, n_channels=16, kernel_size=5, strides=1, padding='valid')
    digitCaps = CapsuleLayer(num_capsule=n_class, dim_vector=8, num_routing=num_routing, name='DigitCaps')(primarycaps)
    out = Length(name='CapsNet')(digitCaps)
    train_model = Model(x, out)
    return train_model


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))



model= CapsNet(input_shape=(64,1,3),n_class=2,num_routing=3)
model.summary()

kf= KFold(n_splits=5, shuffle=True)

tprs = []
mean_fpr = np.linspace(0, 1, 100)
y_real = []
y_proba = []
AverageResult = []


for i,(train_index,test_index) in enumerate(kf.split(sample_data)):
    x_train, x_test=sample_data[train_index],sample_data[test_index]
    y_train, y_test=sample_label[train_index],sample_label[test_index]
    y_train=np.asarray(y_train)

    x_test_temp=SampleFeature_original[test_index]
    y_test_temp=SampleLabel_original[test_index]

    save_x_test=np.array(x_test_temp)
    np.savetxt('Feature(Dataset1)(test)'+str(i+1)+'.txt',save_x_test,fmt='%f',delimiter=' ')
    save_y_test=np.array(y_test_temp)
    np.savetxt('Label(Dataset1)(test)'+str(i+1)+'.txt',save_y_test,fmt='%f',delimiter=' ')

    temp1=len(x_train)
    temp2=int(0.8*temp1)

    x_train_new=x_train[0:temp2-1]
    x_valid=x_train[temp2:temp1-1]

    y_train_new=y_train[0:temp2-1]
    y_valid=y_train[temp2:temp1-1]


    model= CapsNet(input_shape=(64,1,3),n_class=2,num_routing=3)
    model.compile(optimizers.RMSprop(lr=0.0001),loss=margin_loss,metrics=['acc'])
    call=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)]
    
    model.fit(x=x_train_new,y=y_train_new,batch_size=128,epochs=500,validation_data=[x_valid,y_valid], callbacks=call)
    model.save('CapsNet-LDA_model'+str(i+1)+'(Dataset1).h5')

    y_test_pred = model.predict(x_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test_temp, y_test_pred)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    precision, recall, _ = precision_recall_curve(y_test_temp, y_test_pred)
    y_real.append(y_test_temp)
    y_proba.append(y_test_pred)

    y_test_pred2=[]
    for i in range(len(y_test_pred)):
        if y_test_pred[i]>=0.5:
            y_test_pred2.append(1)
        else:
            y_test_pred2.append(0)

    Result = MyConfusionMatrix(y_test_temp,y_test_pred2)
    AverageResult.append(Result)


MyAverage(AverageResult)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
save_fpr=np.array(mean_fpr)
np.savetxt('CapsNet-LDA_fpr(Dataset1)(5-CV).txt',save_fpr,fmt='%f',delimiter=' ')
save_tpr=np.array(mean_tpr)
np.savetxt('CapsNet-LDA_tpr(Dataset1)(5-CV).txt',save_tpr,fmt='%f',delimiter=' ')
mean_auc = auc(mean_fpr, mean_tpr)   
print('CapsNet-LDA_auc(5-CV):')
print(mean_auc)

plt.plot(mean_fpr, mean_tpr, label=r'CapsNet-LDA (AUC = %0.4f)' % (mean_auc),
             lw=1, alpha=.8, color='#FF0000')

plt.xlabel('False Posotive Rate', fontsize=13)
plt.ylabel('True Posotive Rate', fontsize=13)
plt.legend(loc="lower right")
plt.savefig('auc(CapsNet-LDA)(Dataset1).png', dpi=300)
plt.show()

plt.figure(2)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
print('CapsNet-LDA_aupr(5-CV):')
print(auc(recall, precision))
save_precision=np.array(precision)
np.savetxt('CapsNet-LDA_precision(Dataset1)(5-CV).txt',save_precision,fmt='%f',delimiter=' ')
save_recall=np.array(recall)
np.savetxt('CapsNet-LDA_recall(Dataset1)(5-CV).txt',save_recall,fmt='%f',delimiter=' ')

mean_aupr = auc(recall, precision)
plt.plot(recall, precision, label=r'CapsNet-LDA (AUPR = %0.4f)' % (mean_aupr),
             lw=1, alpha=.8, color='#FF0000')

plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
plt.legend(loc="lower left")
plt.savefig('aupr(CapsNet-LDA)(Dataset1).png', dpi=300)
plt.show()



