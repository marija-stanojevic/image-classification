from __future__ import division, print_function, absolute_import

import tflearn
import batches as b
import pickle
from layers.residual_block import residual_block

def buildNetwork(n):
    # Real-time data preprocessing
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    # Building Residual Network
    net = tflearn.input_data(shape=[None, b.IMG_WIDTH, b.IMG_HEIGHT, b.CHANNELS], data_preprocessing=img_prep,
                             data_augmentation=img_aug)
    net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)

    # resnet part
    net = residual_block(net, n, 16, downsample=True)
    net = residual_block(net, 1, 32, downsample=True)
    net = residual_block(net, n - 1, 32, downsample=True)
    net = residual_block(net, 1, 64, downsample=True)
    net = residual_block(net, n - 1, 64)

    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    # Regression
    net = tflearn.fully_connected(net, b.CLASS_3_NUMBER, activation='softmax')
    mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
    net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=b.PATH + 'log')
    return model

def train(n, firstRun, epochs_no, starting_epoch_no, last_model_file = None):
    if firstRun:
        total_training_batches_no = b.divide_into_files(True)
        total_test_batches_no = b.divide_into_files(False)
    else:
        total_training_batches_no = 707
        total_test_batches_no = 177

    cvX, cvY1, cvY2, cvY3 = b.load_cross_validation()
    # put appropriate Y value here depending on the category you want to run for
    cvY = tflearn.data_utils.to_categorical(cvY3, b.CLASS_3_NUMBER)

    # Training
    model = buildNetwork(n)
    if (starting_epoch_no > 0):
        model.load(b.PATH + last_model_file)
    for j in range(starting_epoch_no, epochs_no):
        validation_set = None
        snapshot_epoch = False
        for i in range(0, total_training_batches_no):
            X, Y, Y1, Y2 = b.load_batch(i, True)
            Y = tflearn.data_utils.to_categorical(Y, b.CLASS_3_NUMBER)
            if i == total_training_batches_no - 1 or i % 50 == 0:
                validation_set = cvX, cvY
                snapshot_epoch = True
            model.fit(X, Y, n_epoch=1, validation_set=validation_set, snapshot_epoch=snapshot_epoch, show_metric=True,
                      run_id='resnet')
            if i % 50 == 0:
                model.save(b.PATH + 'ALLmodel_resnet_batch_' + str(i) + 'epoch_' + str(j) + '.txt')

        model.save(b.PATH + 'ALLmodel_resnet_epoch_' + str(j) + '.txt')
    return [model, total_test_batches_no]

def test(model, total_test_batches_no):
    predictions = []
    for i in range(0, total_test_batches_no):
        testX, product_id, Y3, Y4 = b.load_batch(i, False)
        size = len(testX) / 10
        for i in range(0, 10):
            testXpart = testX[i * size, min((i + 1) * size, len(testX))]
            predictions.extend(model.predict_label(testXpart))
    with open(b.PATH + 'ALL' + 'resNetPredictions.txt', 'wb') as pickleFile:
        pickle.dump(predictions, pickleFile)

def main():
    # Number of resisdual blocks
    n = 4
    epochs_no = 100
    starting_epoch_no = 0
    # last epoch model saved location should be provided below (only if starting_epoch_no > 0)
    # last_model_file = 'resNet707Batches/ALLmodel_resnet_epoch_4.txt'
    # in first run it divides dataset into batches, otherwise
    firstRun = True
    model, total_test_batches_no = train(n, firstRun, epochs_no, starting_epoch_no)
    test(model, total_test_batches_no)

if __name__ == "__main__":
    main()
