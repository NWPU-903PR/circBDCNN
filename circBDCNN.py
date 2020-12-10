import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense, MaxPooling1D, Flatten, Dropout, AveragePooling1D, add
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from gensim.models import Word2Vec
from keras.layers import Embedding, Convolution1D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from sklearn.externals import joblib
import multiprocessing
import random
from keras.utils import np_utils, generic_utils
from keras.models import Model
from sklearn.preprocessing import LabelEncoder
import pysam
import argparse
from sklearn import metrics
from gensim.models.word2vec import LineSentence
import pickle
import numpy as np
import gensim, logging
from keras.models import load_model
from keras.layers import concatenate
from sklearn.metrics import matthews_corrcoef,f1_score
import keras.backend.tensorflow_backend as KTF
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
KTF.set_session(sess)

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def seq2ngram(seqs, k, s, dest, wv):
    f = open(seqs)
    lines = f.readlines()
    f.close()
    list22 = []
    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')

    for num, line in enumerate(lines):
        if num < 200000:
            line = line[:-1].lower()  # remove '\n' and lower ACGT
            l = len(line)  # length of line
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
            list22.append(convert_data_to_index(list2, wv))
    f.close()
    return list22


def seq2ngram2(seqs, k, s, dest):
    f = open(seqs)
    lines = f.readlines()
    f.close()

    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')
    for num, line in enumerate(lines):
        if num < 100000:
            line = line[:-1].lower()  # remove '\n' and lower ACGT
            l = len(line)  # length of line

            for i in range(0, l, s):
                if i + k >= l + 1:
                    break

                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
    f.close()


def word2vect(k, s, vector_dim, root_path, pos_sequences):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    seq2ngram2(pos_sequences, k, s, 'seq_pos_' + str(k) + '_' + str(s) + '.txt')
    sentences = LineSentence('seq_pos_' + str(k) + '_' + str(s) + '.txt')

    mode1 = gensim.models.Word2Vec(sentences, iter=20, window=int(18 / s), min_count=50, size=vector_dim,
                                   workers=multiprocessing.cpu_count())
    mode1.save(root_path + 'word2vec_model' + '_' + str(k) + '_' + str(s) + '_' + str(vector_dim))


def build_class_file(np, ng, class_file):
    with open(class_file, 'w') as outfile:
        outfile.write('label' + '\n')
        for i in range(np):
            outfile.write('1' + '\n')
        for i in range(ng):
            outfile.write('0' + '\n')


def dilanet(inputs, filter_nums, filter_size, pad_stride, dilate):

    first_cov = Convolution1D(nb_filter=filter_nums, filter_length=filter_size,
                              activation='relu', dilation_rate=dilate)(inputs)
    first_cov = Convolution1D(nb_filter=1, filter_length=1, use_bias=False)(first_cov)
    first_cov = MaxPooling1D(pad_stride, pad_stride)(first_cov)
    # first_cov = Dropout(0.1)(first_cov)

    return first_cov


def pooling_net(inputs, pad_stride):


    pool1 = AveragePooling1D(pad_stride, pad_stride)(inputs)
    pool2 = MaxPooling1D(pad_stride, pad_stride)(inputs)
    add_out = add([pool1, pool2])
    # add_out = Dropout(0.1)(add_out)

    return add_out


def BDCNN_double(input_shape, embedding_matrix, filter_nums, filter_size, pad_stride, dilate, layers):
    j = 1
    dil = []
    for i in range(layers):
        if j > dilate:
            j = 1
        dil.append(j)
        j += 1
    inputs = Input(shape=input_shape)
    inputs1 = Input(shape=input_shape)
    embedout = Embedding(input_dim=embedding_matrix.shape[0],

              output_dim=embedding_matrix.shape[1],

              weights=[embedding_matrix],

              input_length=input_shape[0],

              trainable=True, name='embedout')(inputs)

    embedout1 = Embedding(input_dim=embedding_matrix.shape[0],

              output_dim=embedding_matrix.shape[1],

              weights=[embedding_matrix],

              input_length=input_shape[0],

              trainable=True, name='embedout1')(inputs1)

    net_train = []
    for i in range(layers):
        net_train.append(dilanet(embedout, filter_nums, filter_size, pad_stride, dil[i]))
    merge_one0 = concatenate(net_train, axis=1)

    net_train1 = []
    for i in range(layers):
        net_train1.append(dilanet(embedout1, filter_nums, filter_size, pad_stride, dil[i]))
    merge_one1 = concatenate(net_train1, axis=1)

    merge_one = concatenate([merge_one0, merge_one1], axis=1)
    # merge_one = Dropout(0.5)(merge_one)
    merge_one = Flatten()(merge_one)
    merge_one = BatchNormalization()(merge_one)
    merge_one = Dropout(0.25)(merge_one)

    dense1 = Dense(20, activation='relu', name='myfeatures')(merge_one)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.25)(dense1)
    dense2 = Dense(2, activation='softmax')(dense1)

    model = Model(inputs=[inputs, inputs1],  outputs=dense2)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                   metrics=['accuracy'])

    print(model.summary())

    return model


def nn_cons_model(input_shape, pad, layers):
    pad_stride = list(range(pad, pad+layers))
    inputs = Input(shape=input_shape)
    net = pooling_net(inputs, pad_stride[0])
    net1 = pooling_net(inputs, pad_stride[1])
    merge_one = concatenate([net, net1], axis=1)
    for layer in range(2, layers):
        net2 = pooling_net(inputs, pad_stride[layer])
        merge_one = concatenate([merge_one, net2], axis=1)

    merge_one = Dropout(0.1)(merge_one)
    merge_one = Flatten()(merge_one)

    dense1 = Dense(20, activation='relu', name='myfeature')(merge_one)
    dense1 = Dropout(0.1)(dense1)
    dense2 = Dense(2, activation='softmax')(dense1)

    model = Model(inputs=inputs,  outputs=dense2)

    model.compile(optimizer='adam', loss='mean_squared_error',
                   metrics=['accuracy'])

    print(model.summary())
    return model


def extract_BDCNN(k, s, vector_dim, MAX_LEN, pos_sequences, neg_sequences, seq_file, class_file,model_dir):
    model1 = gensim.models.Word2Vec.load(
        model_dir + 'word2vec_model' + '_' + str(k) + '_' + str(s) + '_' + str(vector_dim))

    pos_list = seq2ngram(pos_sequences, k, s, 'seq_pos_' + str(k) + '_' + str(s) + '.txt', model1.wv)
    with open(str(k) + '_' + str(s) + 'listpos.pkl', 'wb') as pickle_file:
        pickle.dump(pos_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    neg_list = seq2ngram(neg_sequences, k, s, 'seq_neg_' + str(k) + '_' + str(s) + '.txt', model1.wv)
    with open(str(k) + '_' + str(s) + 'listneg.pkl', 'wb') as pickle_file:
        pickle.dump(neg_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    seqs = pos_list + neg_list

    X = pad_sequences(seqs, maxlen=int(MAX_LEN/2))
    XX = pad_sequences(seqs, maxlen=int(MAX_LEN/2), truncating='post')
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))
    build_class_file(len(pos_list), len(neg_list), class_file)
    X1 = X
    X2 = XX

    n_seqs = len(seqs)
    indices = np.arange(n_seqs)

    np.random.shuffle(indices)

    X = X[indices]
    XX = XX[indices]

    y = y[indices]

    n_tr = int(n_seqs * 0.8)

    X_train = X[:n_tr]
    X_train_post = XX[:n_tr]

    y_train = y[:n_tr]

    X_valid = X[n_tr:]
    X_valid_post = XX[n_tr:]

    y_valid = y[n_tr:]

    embedding_matrix = np.zeros((len(model1.wv.vocab), vector_dim))
    for i in range(len(model1.wv.vocab)):
        embedding_vector = model1.wv[model1.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    input_shape = X_train[0].shape
    filter_nums, filter_size = 10, 7
    pad_stride = 2
    dilate = 3
    layers = 3
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_valid = to_categorical(y_valid, num_classes)

    model = BDCNN_double(input_shape, embedding_matrix, filter_nums, filter_size, pad_stride, dilate, layers)

    checkpointer = ModelCheckpoint(
        filepath=model_dir+'bestmodel_doubel_bdcnn' + str(k) + ' ' + str(s) + ' ' + str(vector_dim) + str(MAX_LEN) + '.hdf5',
        verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print('Training model...')
    history = model.fit([X_train, X_train_post], y_train, nb_epoch=20, batch_size=128, shuffle=True,
                        validation_data=([X_valid, X_valid_post], y_valid),
                        callbacks=[checkpointer, earlystopper],
                        verbose=2)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('myfeatures').output)

    np.savetxt(seq_file, intermediate_layer_model.predict([X1, X2]), delimiter=" ")


def extract_NN_cons(data_dir,model_dir,file_circ,file_lnc,outfile):

    model_path = model_dir + 'best_model_cons.hdf5'

    datafile_c = np.load(file_circ)
    cons_score_circ = datafile_c['cons_score']
    datafile_l = np.load(file_lnc)
    cons_score_lnc = datafile_l['cons_score']
    label = np.array([1] * len(cons_score_circ) + [0] * len(cons_score_lnc))
    data = np.vstack((cons_score_circ, cons_score_lnc))

    original_train_data = data
    original_label = label

    n_seqs = len(data)
    indices = np.arange(n_seqs)
    np.random.shuffle(indices)

    data = data[indices]
    label = label[indices]

    n_tr = int(n_seqs * 0.8)

    train_data = data[:n_tr]
    train_label = label[:n_tr]
    val_data = data[n_tr:]
    val_label = label[n_tr:]
    num_classes = 2

    input_shape = train_data[0].shape

    pad = 4
    layers = 3

    train_label = to_categorical(train_label, num_classes)
    val_label = to_categorical(val_label, num_classes)

    model = nn_cons_model(input_shape, pad, layers)

    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(train_data, train_label, batch_size=128, epochs=20, verbose=2, shuffle=True,
              callbacks=[checkpointer, earlystop], validation_data=(val_data, val_label))

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('myfeature').output)

    np.savetxt(outfile, intermediate_layer_model.predict(original_train_data), delimiter=" ")


def extract_features_training(pos_sequences, neg_sequences, data_dir,model_dir, seq_file, cons_file,class_file):

    word2vect(3, 1, 40, model_dir, pos_sequences)
    print('extract_BDCNN')
    extract_BDCNN(3, 1, 40, 8000, pos_sequences, neg_sequences, seq_file, class_file,model_dir)

    print('extract_NN_cons')
    file_circ = data_dir + 'circRNA_cons_test.npz'
    file_lnc = data_dir + 'lncRNA_cons_test.npz'
    extract_NN_cons(data_dir, model_dir,file_circ,file_lnc,cons_file)

    print('extract feature done')
    return cons_file, seq_file


def fuse_feature(seq_file, cons_file, output,data_dir,model_dir):

    f = open(output, 'w')

    model_path_fin = model_dir + 'fuse_model.hdf5'

    training_data = load_data(data_dir, cons_file, seq_file)

    seq_hid = 40

    cons_hid = 60
    training_indice, training_label, validation_indice, validation_label, test_indice, testing_label = split_training_validation_test(
        training_data["Y"], shuffle=True)
    print('split done')
    model_input = []

    print(np.shape(training_data["seq"]))

    seq_data, seq_scaler = preprocess_data(training_data["seq"])
    print(np.shape(seq_data))
    joblib.dump(seq_scaler, os.path.join(model_dir, 'seq_scaler1.pkl'))

    seq_train = seq_data[training_indice]

    seq_validation = seq_data[validation_indice]

    seq_test = seq_data[test_indice]
    inputt = Input(shape=(seq_train.shape[1],))
    model_input.append(inputt)
    # seq_net = get_rnn_fea(inputt, sec_num_hidden=seq_hid, num_hidden=seq_hid * 2)
    seq_net = get_rnn_fea(inputt, sec_num_hidden=15, num_hidden=10)

    seq_data = []

    training_data["seq"] = []


    cons_data, cons_scaler = preprocess_data(training_data["cons"])

    joblib.dump(cons_scaler, os.path.join(model_dir, 'cons_scaler1.pkl'))

    cons_train = cons_data[training_indice]
    cons_test = cons_data[test_indice]
    cons_validation = cons_data[validation_indice]
    inputt = Input(shape=(cons_train.shape[1],))
    model_input.append(inputt)

    cons_net = get_rnn_fea(inputt, sec_num_hidden=40, num_hidden=60)

    cons_data = []

    training_data["cons"] = []

    train_y, encoder = preprocess_labels(training_label)

    val_y, encoder = preprocess_labels(validation_label, encoder=encoder)
    test_y, encoder = preprocess_labels(testing_label, encoder=encoder)
    training_data.clear()


    training_net = []
    training = []
    testing = []
    validation = []
    total_hid = 0


    training_net.append(seq_net)

    training.append(seq_train)

    validation.append(seq_validation)
    testing.append(seq_test)

    total_hid = total_hid + seq_hid

    seq_train = []

    seq_validation = []

    training_net.append(cons_net)

    training.append(cons_train)

    validation.append(cons_validation)
    testing.append(cons_test)

    total_hid = total_hid + cons_hid

    cons_train = []

    cons_validation = []

    merge_one = concatenate(training_net)

    merge_one = Dense(10, input_shape=(total_hid,), activation='relu')(merge_one)
    merge_one = Dropout(0.1)(merge_one)
    merge_one = Dense(2, activation='softmax')(merge_one)

    model = Model(inputs=model_input, outputs=merge_one)
    sgd = optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    checkpointer = ModelCheckpoint(filepath=model_path_fin, verbose=1, save_best_only=True)

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print('model training')

    model.fit(training, train_y, batch_size=128, nb_epoch=20, verbose=2, validation_data=(validation, val_y),
              callbacks=[checkpointer, earlystopper])

    model.save(os.path.join(model_dir, 'bestmodel_BDCNN_FUSE_padcons.pkl'))

    model_best = load_model(model_path_fin)
    train_loss, train_acc = model_best.evaluate(training, train_y)
    test_loss, test_acc = model_best.evaluate(testing, test_y)

    train_prob = model_best.predict(training)
    test_prob = model_best.predict(testing)
    train_pred_label = np.argmax(train_prob, axis=1)
    test_pred_label = np.argmax(test_prob, axis=1)
    train_MCC = matthews_corrcoef(training_label, train_pred_label)
    test_MCC = matthews_corrcoef(testing_label, test_pred_label)
    train_F1 = f1_score(training_label, train_pred_label)
    test_F1 = f1_score(testing_label, test_pred_label)

    f.write('train: ACC:' + str(train_acc)[:6] + ', MCC:' + str(train_MCC)[:6] + ', F1-score:' + str(train_F1)[:6] + '\n')
    f.write('test: ACC:' + str(test_acc)[:6] + ', MCC:' + str(test_MCC)[:6] + ', F1-score:' + str(test_F1)[:6] + '\n')


def load_data(path, cons_file=None, seq_file=None):
    """

        Load data matrices from the specified folder.

    """

    data = dict()

    data["seq"] = np.loadtxt(seq_file, delimiter=' ', skiprows=0)

    data["cons"] = np.loadtxt(cons_file, delimiter=' ', skiprows=0)

    data["Y"] = np.loadtxt(path + 'class.txt', skiprows=1)

    print('data loaded')

    return data


def preprocess_data(X, scaler=None, stand=False):
    if not scaler:

        if stand:

            scaler = StandardScaler()

        else:

            scaler = MinMaxScaler()

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()

        encoder.fit(labels)

    y = encoder.transform(labels).astype(np.int32)

    if categorical:
        y = np_utils.to_categorical(y)

    return y, encoder


def get_rnn_fea(train, sec_num_hidden=128, num_hidden=128):
    # inputs = Input(shape=(train.shape[1],))

    firs = Dense(num_hidden, activation='relu')(train)
    firs = PReLU()(firs)
    firs = BatchNormalization()(firs)
    firs = Dropout(0.1)(firs)
    firs = Dense(sec_num_hidden, input_dim=sec_num_hidden, activation='relu')(firs)
    firs = PReLU()(firs)
    firs = BatchNormalization()(firs)
    firs = Dropout(0.1)(firs)


    # model = Sequential()
    #
    # # model.add(Dense(num_hidden, input_dim=train.shape[1], activation='relu'))
    #
    # model.add(Dense(num_hidden, input_shape=(train.shape[1],), activation='relu'))
    #
    # model.add(PReLU())
    #
    # model.add(BatchNormalization())
    #
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(sec_num_hidden, input_dim=sec_num_hidden, activation='relu'))
    #
    # # model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))
    #
    # model.add(PReLU())
    #
    # model.add(BatchNormalization())
    #
    # # model.add(Activation('relu'))
    #
    # model.add(Dropout(0.5))

    return firs


def split_training_validation_test(classes, train_rate=0.75, val_rate=0.1, test_rate=0.15, shuffle=False):
    """split sampels based on balnace classes"""

    num_samples = len(classes)

    classes = np.array(classes)

    classes_unique = np.unique(classes)

    num_classes = len(classes_unique)

    indices = np.arange(num_samples)

    # indices_folds=np.zeros([num_samples],dtype=int)

    training_indice = []

    training_label = []
    test_indice = []

    test_label = []

    validation_indice = []

    validation_label = []

    print(str(classes_unique))
    for cl in classes_unique:

        indices_cl = indices[classes == cl]

        num_samples_cl = len(indices_cl)

        # split this class into k parts

        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual
        num_samples_train = int(num_samples_cl * train_rate)
        num_samples_validation = int(num_samples_cl * val_rate)
        res = num_samples_cl - num_samples_validation - num_samples_train
        training_indice = training_indice + [val for val in indices_cl[:num_samples_train]]
        training_label = training_label + [cl] * num_samples_train

        validation_indice = validation_indice + [val for val in indices_cl[num_samples_train:num_samples_train+num_samples_validation]]
        validation_label = validation_label + [cl] * num_samples_validation
        test_indice = test_indice + [val for val in indices_cl[num_samples_train+num_samples_validation:]]
        test_label = test_label + [cl] * res

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]

    test_index = np.arange(len(test_label))
    random.shuffle(test_index)
    test_indice = np.array(test_indice)[test_index]
    test_label = np.array(test_label)[test_index]

    print(np.shape(training_indice))
    print(np.shape(training_label))
    print(np.shape(validation_indice))
    print(np.shape(validation_label))
    print(np.shape(test_indice))
    print(np.shape(test_label))

    return training_indice, training_label, validation_indice, validation_label, test_indice, test_label


def train_circBDCNN(data_dir, model_dir, positive_data, negative_data, output_file,extract_features=True):

    cons_file = data_dir + 'conservation_features.txt'
    seq_file = data_dir + 'seq_features.txt'
    class_file = data_dir + 'class.txt'

    if extract_features:
        print('extract_features')
        cons_file, seq_file = extract_features_training(positive_data, negative_data, data_dir, model_dir, seq_file, cons_file,class_file)

    fuse_feature(seq_file, cons_file, output_file,data_dir,model_dir)

    print('ok')


def run_circBDCNN(parser):

    data_dir = parser.data_dir

    out_file = parser.out_file

    model_dir = parser.model_dir

    positive_data = args.positive_data
    negative_data = args.negative_data

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print('model training')

    train_circBDCNN(data_dir, model_dir, positive_data, negative_data, out_file, True)


def parse_arguments(parser):
    parser.add_argument('--data_dir', type=str, default='data/', metavar='<data_directory>')

    parser.add_argument('--model_dir', type=str, default='models/')

    parser.add_argument('--out_file', type=str, default='prediction.txt')

    parser.add_argument('--positive_data', type=str, default='data/circRNA_dataset_test.txt')

    parser.add_argument('--negative_data', type=str, default='data/lncRNA_dataset_test.txt')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='circBDCNN: distinguishing circular RNAs from other long non-coding RNAs with broad dilation convolutional neural network')

    args = parse_arguments(parser)

    run_circBDCNN(args)
