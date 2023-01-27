import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def contrastive_loss_with_margin(margin):
    """Provides 'contrastive_loss' with an enclosing scope with variable 'margin'.
    :param margin: integer defining the baseline for distance for which pairs should be classified as dissimilar
    :returns: 'contrastive_loss' function with a margin attached
    """

    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.
        :param y_true: list of labels, each label is of type float32
        :param y_pred: list of predictions of same lengths as of y_true, each label is of type float32
        :returns: a tensor containing contrastive loss as floating point value
        """
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))

        return (y_true * square_pred + (1 - y_true) * margin_square)

    return contrastive_loss


def performance(cm):
    """Calculate a set of test performance indicators.
    :param cm: 2x2 confusion matrix of type numpy.array
    :returns: list of 6 float32 performance indicators as a percentage: accuracy, sensitivity, specificity, precision, recall and F1-score, in that order
    """
    Acc = (cm[0, 0]+cm[1, 1])/(cm[0, 0]+cm[0, 1]+cm[1, 0]+cm[1, 1])
    Sen = cm[0, 0]/(cm[0, 0]+cm[1, 0])
    Spe = cm[1, 1]/(cm[1, 1]+cm[0, 1])
    Pre = cm[0, 0]/(cm[0, 0]+cm[0, 1])
    Rec = cm[0, 0]/(cm[0, 0]+cm[1, 0])
    F = (2*Pre*Rec)/(Pre+Rec)

    return [Acc*100, Sen*100, Spe*100, Pre*100, Rec*100, F*100]


def euclidean_distance(vects):
    """Calculates the Euclidean distance metric to measure the similarity between the images.
    :param vects: Feature vector 
    :returns: Euclidean distance 
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):

    shape1, shape2 = shapes
    return (shape1[0], 1)


def tensor_pair(Path_Ch_1, Path_Ch_2, height, width, label):
    """ Builds a pair of tensors based on two folders containing chromosome images.
    :param Path_Ch_1: string, path of a folder containing pair 1 chromosome images
    :param Path_Ch_2: string, path of a folder containing pair 2 chromosome images
    :param height: integer, height of the images in number of pixels
    :param width: integer, width of the images in number of pixels
    :param label: vector of label (0 for normal and 1 for deletion)
    :returns: tensors of prepared data
    """
    Data_Ch_1, Data_Ch_2 = [], []
    Label = []

    for i, j in zip(sorted(os.listdir(Path_Ch_1)), sorted(os.listdir(Path_Ch_2))):
        if i.endswith('TIF'):

            array_5_1 = np.array(Image.open(Path_Ch_1+i), dtype=np.uint8)
            array_5_2 = np.array(Image.open(Path_Ch_2+j), dtype=np.uint8)

            array_5_1 = array_5_1.reshape(
                array_5_1.shape[0], array_5_1.shape[1], 1)
            array_5_2 = array_5_2.reshape(
                array_5_2.shape[0], array_5_2.shape[1], 1)

            array_5_1 = cv2.cvtColor(array_5_1, cv2.COLOR_GRAY2BGR)
            array_5_2 = cv2.cvtColor(array_5_2, cv2.COLOR_GRAY2BGR)

            array_5_1 = cv2.resize(
                array_5_1, (width, height), interpolation=cv2.INTER_AREA)
            array_5_2 = cv2.resize(
                array_5_2, (width, height), interpolation=cv2.INTER_AREA)

            Data_Ch_1.append(array_5_1)
            Data_Ch_2.append(array_5_2)

            Label.append(label)

    Data_Ch_1 = np.array(Data_Ch_1)
    Data_Ch_2 = np.array(Data_Ch_2)
    Label = np.array(Label)

    return Data_Ch_1, Data_Ch_2, Label


def train_and_test(Path_Ch_1_Normal, Path_Ch_2_Normal, Path_Ch_1_Abnormal, Path_Ch_2_Abnormal, height, width, channels, num_Epochs, m, model_name):
    """ Performs training and test based the Siamese architecture.
    :param Path_Ch_1_Normal: string, path of the folder containing pair 1 images of normal chromosome pairs
    :param Path_Ch_2_Normal: string, path of the folder containing pair 2 images of normal chromosome pairs
    :param Path_Ch_1_Abnormal: string, path of the folder containing pair 1 images of abnormal chromosome pairs
    :param Path_Ch_2_Abnormal: string, path of the folder containing pair 2 images of abnormal chromosome pairs
    :param height: integer, height of the images in number of pixels
    :param width: integer, width of the images in number of pixels
    :param channels: integer, number of channels of the considered images
    :param num_Epochs: integer, number of epochs for the training phase
    :param m: integer, chosen margin for the contrastive loss function
    :param model_name: string, chosen CNN model (e.g. 'MobileNet', etc)
    """

    # Data loading phase
    print('Loading Data ...')

    Data_Ch_1_Normal, Data_Ch_2_Normal, Label_Normal = tensor_pair(
        Path_Ch_1_Normal, Path_Ch_2_Normal, height, width, 0)
    print('Normal pair = ', len(Data_Ch_1_Normal))

    Data_Ch_1_Abnormal, Data_Ch_2_Abnormal, Label_Abnormal = tensor_pair(
        Path_Ch_1_Abnormal, Path_Ch_2_Abnormal, height, width, 1)
    print('Abnormal pair = ', len(Data_Ch_1_Abnormal))

    Data_Chr_1 = np.concatenate((Data_Ch_1_Normal, Data_Ch_1_Abnormal), axis=0)
    Data_Chr_2 = np.concatenate((Data_Ch_2_Normal, Data_Ch_2_Abnormal), axis=0)
    Label = np.concatenate((Label_Normal, Label_Abnormal), axis=0)

    # Training/test phase
    Performance = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    f = -1

    for train_index, test_index in kfold.split(Data_Chr_1, Label):
        f = f+1

        Train_Ten_1 = (Data_Chr_1[train_index]).astype('float32')
        Train_Ten_2 = (Data_Chr_2[train_index]).astype('float32')
        Label_Train = (Label[train_index]).astype('float32')

        Valid_Ten_1 = (Data_Chr_1[test_index]).astype('float32')
        Valid_Ten_2 = (Data_Chr_2[test_index]).astype('float32')
        Label_Valid = (Label[test_index]).astype('float32')

        model_choice = getattr(tf.keras.applications, model_name)
        CNN1 = model_choice(include_top=False, weights="imagenet",
                            input_shape=(height, width, channels))
        CNN2 = model_choice(include_top=False, weights="imagenet",
                            input_shape=(height, width, channels))

        for i, layer in enumerate(CNN1.layers):
            layer._name = 'CNN1_' + str(i)
        for i, layer in enumerate(CNN2.layers):
            layer._name = 'CNN2_' + str(i)

        model1 = Flatten(name='flatten_1')(CNN1.output)
        model2 = Flatten(name='flatten_2')(CNN2.output)
        output = Lambda(euclidean_distance, name='output_layer',
                        output_shape=eucl_dist_output_shape)([model1, model2])
        model = Model(inputs=[CNN1.input, CNN2.input],
                      outputs=output, name='Xception')

        for layer in model.layers:
            layer.trainable = True

        optimizers = tf.keras.optimizers.RMSprop()
        model.compile(loss=contrastive_loss_with_margin(
            margin=m), optimizer=optimizers)
        checkpoint = ModelCheckpoint(
            'Best_Model.hdf5', monitor='val_loss', verbose=1, save_weights_only=True, mode='auto')
        model.fit(x=[Train_Ten_1, Train_Ten_2],
                  y=Label_Train,
                  epochs=num_Epochs,
                  callbacks=[checkpoint],
                  batch_size=32,
                  validation_data=([Valid_Ten_1, Valid_Ten_2], Label_Valid))

        model.load_weights('Best_Model.hdf5')

        # Performance measure
        y_pred = model.predict([Valid_Ten_1, Valid_Ten_2])

        y_pred = y_pred.ravel() > 0.5

        cm = confusion_matrix(Label_Valid, y_pred)
        Perf = performance(cm)
        matthews = matthews_corrcoef(Label_Valid, y_pred)

        # Append a list containig TP, FP, TN, FN, accuracy, sensitivity, specificity, F1-score and Matthew correlation coefficient
        Performance.append([cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
                           Perf[0], Perf[1], Perf[2], Perf[5], matthews])

    np.savetxt('Results.csv', Performance, fmt='%10.2f', delimiter=',')


if __name__ == "__main__":
    
    # Examples of parameters
    height = 141
    width = 75
    channels = 3
    num_Epochs = 1
    m = 1
    model_name = 'MobileNet'
    Path_Ch_1_Normal = 'C:/Users/isen/Desktop/Post_Doc_Brest/Database_Brest/CHROMAI_Data/Data_For_Test/Chromosome_5_(Data_2)/Deletion_Chromosome_5/Ch_5_1/'
    Path_Ch_2_Normal = 'C:/Users/isen/Desktop/Post_Doc_Brest/Database_Brest/CHROMAI_Data/Data_For_Test/Chromosome_5_(Data_2)/Deletion_Chromosome_5/Ch_5_1/'
    Path_Ch_1_Abnormal = 'C:/Users/isen/Desktop/Post_Doc_Brest/Database_Brest/CHROMAI_Data/Data_For_Test/Chromosome_5_(Data_2)/Deletion_Chromosome_5/Ch_5_1/'
    Path_Ch_2_Abnormal = 'C:/Users/isen/Desktop/Post_Doc_Brest/Database_Brest/CHROMAI_Data/Data_For_Test/Chromosome_5_(Data_2)/Deletion_Chromosome_5/Ch_5_1/'

    # Usage example of function train_and_test
    train_and_test(Path_Ch_1_Normal, Path_Ch_2_Normal, Path_Ch_1_Abnormal,
                   Path_Ch_2_Abnormal, height, width, channels, num_Epochs, m, model_name)
