# -*- coding: utf-8 -*-
"""
The multi-task model extended the VGG16 convolutional base with three fully connected layers.
The model has two outputs with different network heads: one head is the classification output,
the other represents the annotation. For the binary classification task, again a cross-entropy loss and
sigmoid activation function of the nodes are used. The contribution of the different losses are equal.
The resulting loss values are summed and minimised during network training.
"""
## imports
from LCR_models import VGG16_multi
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from matplotlib import pyplot as plt
import pandas as pd
import os
from LCR_get_data import get_multi_task_data, load_nodule_multi_task
from sklearn.metrics import roc_auc_score, roc_curve
import time

PYTHON_VERSION = '3.6.10'
TENSOR_FLOW_GPU = '2.1.0'

## gpu and log settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

## variables
path = r'/mnt/beta/tjwgoedings/BEPdir/LCR'  #server_path
ground_truth_file = r'/mnt/beta/tjwgoedings/BEPdir/LCR/new_data/new_nodule_info.xlsx' #path to ground truth file
learning_rates = [0.00001, 0.0005, 0.0006]
EPOCHS = 20
BATCH_SIZE = 8
MODEL_NAME = 'VGG16' #used in the name of logging
savefig = True      #save ROC curve
VERBOSE = True
seeds = [10, 19, 1971, 2000]
annotation = 'subtlety'  #Choises: malignacy, subtlety, internalstructure, calcification, sphericity, margin, lobulation, spiculation, texture
run_name = 'multitask_results'

# ---------
# Main code
# ---------

def append_df_to_excel(history, score, model_name, passed_time, excel_path):
    hist_dic = history.history
    auc = {'auc': score}

    dic_summary = {'Model_name': model_name}
    dic_summary.update(hist_dic)
    dic_summary.update(auc)
    dic_summary.update({'time (s)' : passed_time})
    df = pd.DataFrame(dic_summary)

    isExist = os.path.exists(excel_path)
    if not isExist:
        df.to_excel((excel_path), index=False, header=True)
        return
    df_excel = pd.read_excel(excel_path)
    result = pd.concat([df_excel, df], ignore_index=True)
    result.to_excel(excel_path, index=False)
    return

def predict_model(checkpoint_filepath, model_name, path, savefig, VERBOSE):
    if VERBOSE:
        print(f'Loading best weights model from {checkpoint_filepath}')
    model.load_weights(checkpoint_filepath)

    dataset3 = lambda: load_nodule_multi_task(test_id, test_label, test_ann)
    test_dataset = tf.data.Dataset.from_generator(dataset3, output_types=(tf.float32, {'out_class': tf.int64, 'out_anno': tf.float16}), output_shapes=(tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]), 'out_anno': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    predictions = model.predict(test_dataset)
    y_true = test_label
    scores = predictions[0]
    y_true =y_true.to_numpy()
    y_true = y_true.astype('int')
    df = pd.DataFrame({'id': test_id, 'prediction': scores.tolist(), 'true_label': y_true})
    df.to_excel(os.path.join(path, 'new_data', 'ROC_data', f'{model_name}.xlsx'), index=False, header=True)

    auc = roc_auc_score(y_true, scores)

    cf, ct, treshold = roc_curve(y_true, scores)
    plt.title('Receiver Operating Characteristic')
    plt.plot(cf, ct, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    if savefig:
        plt.savefig(os.path.join(path, 'new_data', 'Plots', f'ROC_AUC_{model_name}.png'), bbox_inches='tight')
    return auc

for i in range(len(seeds)):
    seed= seeds[i]

    for j in range(len(learning_rates)):
        lr = learning_rates[j]

        if VERBOSE:
            print(f'creating model for seed {seed} and batch_size: {BATCH_SIZE} / learning_rate: {lr}')

        model = VGG16_multi(lr, VERBOSE, BATCH_SIZE)

        train_id, train_label, train_ann, valid_id, valid_label, valid_ann, test_id, test_label, test_ann, class_weights = get_multi_task_data(ground_truth_file, seed, annotation, VERBOSE) #train_mask, valid_mask, test_mask

        #data generator
        dataset1 = lambda: load_nodule_multi_task(train_id, train_label, train_ann)
        dataset2 = lambda: load_nodule_multi_task(valid_id, valid_label, valid_ann)

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_dataset = tf.data.Dataset.from_generator(dataset1, output_types=(tf.float32, {'out_class': tf.int64, 'out_anno': tf.float16}), output_shapes = (tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]), 'out_anno': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)
        val_dataset = tf.data.Dataset.from_generator(dataset2, output_types=(tf.float32, {'out_class': tf.int64, 'out_anno': tf.float16}), output_shapes = (tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]), 'out_anno': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)

        # save the model and weights
        model_name = f'{MODEL_NAME}_{seed}_{lr}'
        model_filepath = os.path.join(path, 'new_data', 'models', (model_name + '.json'))
        weights_filepath = os.path.join(path, 'new_data', 'models', (model_name + '_weights.hdf5'))

        model_json = model.to_json() # serialize model to JSON
        with open(model_filepath, 'w') as json_file:
            json_file.write(model_json)

        # define the model checkpoint and Tensorboard callbacks
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', min_delta = 0.001)
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        tensorboard = TensorBoard(os.path.join(path, 'new_data', 'logs', model_name))
        callbacks_list = [early_stopping, checkpoint, tensorboard]

        if VERBOSE:
            print(f'started training with #epochs: {EPOCHS}, batch_size: {BATCH_SIZE} and learning_rate: {lr}')

        start = time.time()
        history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset,
                            callbacks=callbacks_list)
        end = time.time()
        passed_time = end-start
        auc_score = predict_model(weights_filepath, model_name, path, savefig, VERBOSE)

        append_df_to_excel(history, auc_score, model_name, passed_time, os.path.join(path, 'new_data', f'{run_name}.xlsx'))

        del history, dataset1, dataset2, train_dataset, val_dataset
        if VERBOSE:
            print(f'saved {model_name}')

print('Done')
