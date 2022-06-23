from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
import numpy as np
import os

## variables
path = r'/mnt/beta/tjwgoedings/BEPdir/LCR'

def padding3D(array, xx, yy, zz):
    #:param array: numpy array
    #:param xx: desired height
    #:param yy: desirex width
    #:return: padded array

    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - d) // 2
    cc = zz - c - d

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')

def get_baseline_data(ground_truth_file, seed, verbose):
    df_load = pd.read_excel(ground_truth_file)
    df = df_load.drop(df_load[df_load.Sickness == 'Inconclusive'].index)
    class_label = df['Sickness']
    class_id = df['Nodule_name']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weights = {i: class_weights[i] for i in range(2)}

    if verbose:
        print('in train set      = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set       = \n' + str(y_test.value_counts()))

    return (X_train, y_train, X_validate, y_validate, X_test, y_test, class_weights)

def get_multi_task_data(ground_truth_file, seed, annotation, verbose):
    if verbose:
        print('In method get_multi_task_data')

    df_load = pd.read_excel(ground_truth_file)
    df = df_load.drop(df_load[df_load.Sickness == 'Inconclusive'].index)
    class_label = df['Sickness']
    class_id = df['Nodule_name']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    df_load.set_index("Nodule_name", inplace=True)

    Train_list = X_train.tolist()
    ann_list = []
    for name in Train_list:
        ann_row = df_load.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_train = pd.Series(ann_list)

    ID_list = X_validate.tolist()
    ann_list = []
    for name in ID_list:
        ann_row = df_load.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_valid = pd.Series(ann_list)

    ID_list = X_test.tolist()
    ann_list = []
    for name in ID_list:
        ann_row = df_load.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_test = pd.Series(ann_list)

    w = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weights = dict()
    class_weights[0] = w[0]
    class_weights[1] = w[1]

    return (X_train, y_train, annotation_train,
            X_validate, y_validate, annotation_valid,
            X_test, y_test, annotation_test, class_weights)
            #sample_weight_train, sample_weight_valid, sample_weight_test)

def load_nodule_baseline(ID_list, Label_list):
    for i in range(len(ID_list)):
        ID = ID_list.iloc[i]
        y_label = Label_list.iloc[i]
        x_label = np.load(os.path.join(path, 'new_data', 'Images2', f'Patient_{ID[0:4]}', f'img_{ID}.npy'))
        nodule_padded = padding3D(x_label, 128, 128, 64)
        nodule_padded = np.resize(nodule_padded, (128, 128, 64, 1))

        yield nodule_padded, y_label

def load_nodule_multi_task(ID_list, Label_list, Ann_list): #, sample_weight):
    for i in range(len(ID_list)):
        ID = ID_list.iloc[i]

        x_label = np.load(os.path.join(path, 'new_data', 'Images2', f'Patient_{ID[0:4]}', f'img_{ID}.npy'))
        nodule_padded = padding3D(x_label, 128, 128, 64)
        nodule_padded = np.resize(nodule_padded, (128, 128, 64, 1))

        y_label = Label_list.iloc[i]
        annotation = Ann_list.iloc[i]
        annotation = (annotation-min(Ann_list))/(max(Ann_list)-min(Ann_list)) #normalize annotation

        #TODO sample_weight = sample_weight.iloc[i]
        #yield nodule_padded, y_label, annotation
        yield (nodule_padded, {'out_class': np.asarray(y_label), 'out_anno': np.asarray(annotation)})
        #{'out_asymm': np.asarray(sample_weight)}))


