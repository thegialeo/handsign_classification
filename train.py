import  argparse
import mxnet as mx
from mxnet import gluon, init, nd, autograd
from mxnet.gluon import nn, utils, data
import mxnet.gluon.data.vision.transforms as transforms
import matplotlib.pyplot as plt
import math
import time
from mxnet.gluon.model_zoo import vision
import cv2
import numpy as np


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def train(net, clf, trainloader, testloader, criterion, trainer_net, trainer_clf, ctx, batch_size, num_epochs,
          use_mask_and_conture=False):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', ctx)
    loss_hist = []
    train_hist = []
    test_hist = []
    for epoch in range(num_epochs):
        start = time.time()
        for i, (data, label) in enumerate(trainloader):
            data = random_rotation(data)
            if use_mask_and_conture:
                mask_contour = preprocessing(data).as_in_context(ctx)
            transformer = transforms.ToTensor()
            data = transformer(data)
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                if use_mask_and_conture:
                    features = net(data)
                    mask_contour_feat = nd.concat(features, mask_contour, dim=1)
                    output = clf(mask_contour_feat)
                    loss = criterion(output, label)
                else:
                    output = clf(net(data))
                    loss = criterion(output, label)
            loss.backward()
            if epoch > 100:
                trainer_net.step(batch_size)
            trainer_clf.step(batch_size)

            # record loss
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (epoch == 0))
                           else (1 - 0.01) * moving_loss + 0.01 * curr_loss)

        test_acc = evaluate_accuracy(testloader, net, clf, ctx, use_mask_and_conture)
        train_acc = evaluate_accuracy(trainloader, net, clf, ctx, use_mask_and_conture)
        print('epoch {}, loss {:.5f}, train acc {:.5f}, test acc {:.5f}, time {:.1f} sec'.format(
            epoch + 1, moving_loss, train_acc, test_acc, time.time() - start))
        loss_hist.append(moving_loss)
        train_hist.append(train_acc)
        test_hist.append(test_acc)

    # save loss plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(loss_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('average loss', fontsize=14)
    if use_mask_and_conture:
        plt.savefig('loss_mask_contour.png')
    else:
        plt.savefig('loss.png')

    # save train accuracy plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(train_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    if use_mask_and_conture:
        plt.savefig('train_acc_mask_contour.png')
    else:
        plt.savefig('train_accuracy.png')

    # save train accuracy plot
    plt.figure(num=None, figsize=(8, 6))
    plt.plot(test_hist)
    plt.grid(True, which="both")
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accuracy', fontsize=14)
    if use_mask_and_conture:
        plt.savefig('test_acc_mask_contour.png')
    else:
        plt.savefig('test_accuracy.png')

    # save model
    if use_mask_and_conture:
        net.save_parameters("net_mask_contour.params")
        clf.save_parameters("clf_mask_contour.params")
    else:
        net.save_parameters("net.params")
        clf.save_parameters("clf.params")



def evaluate_accuracy(data_iter, net, clf, ctx, use_mask_and_conture=False):
    """Evaluate accuracy of a model on the given data set."""
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iter):
        if use_mask_and_conture:
            mask_contour = preprocessing(data).as_in_context(ctx)
        transformer = transforms.ToTensor()
        data = transformer(data)
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        if use_mask_and_conture:
            features = net(data)
            mask_contour_feat = nd.concat(features.flatten(), mask_contour, dim=1)
            output = clf(mask_contour_feat)
        else:
            output = clf(net(data))
        pred = nd.argmax(output, axis=1)
        acc.update(preds=pred, labels=label)
    return acc.get()[1]

def random_rotation(batch):
    for i in range(batch.shape[0]):
        rot_idx = np.random.randint(4)
        if rot_idx == 0: # no rotation
            batch[i,:,:,:] = batch[i,:,:,:]
        elif rot_idx == 1: # 90 degree rotation
            batch[i,:,:,:] = nd.transpose(batch[i,:,:,:], axes=(1, 0, 2))
            batch[i,:,:,:] = nd.image.flip_left_right(batch[i,:,:,:])
        elif rot_idx == 2: # 180 degree rotation
            batch[i,:,:,:] = nd.image.flip_left_right(batch[i,:,:,:])
            batch[i,:,:,:] = nd.image.flip_top_bottom(batch[i,:,:,:])
        elif rot_idx == 3: # 270 degree rotation
            batch[i,:,:,:] = nd.image.flip_left_right(batch[i,:,:,:])
            batch[i,:,:,:] = nd.transpose(batch[i,:,:,:], axes=(1, 0, 2))
    return batch

def preprocessing(batch):
    np_batch = np.uint8(batch.asnumpy())
    mask = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1), np.uint8)
    contour_img = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2], 1), np.uint8)

    for i in range(batch.shape[0]):

        # skin color map in bgra
        bgra = cv2.cvtColor(np_batch[i,:,:,:], cv2.COLOR_BGR2BGRA)
        lower_mask = cv2.inRange(bgra, (20, 40, 95, 15), (255, 255, 255, 255))
        rg_mask = (bgra[:, :, 2] > bgra[:, :, 1]).astype(np.uint8) * 255
        rb_mask = (bgra[:, :, 2] > bgra[:, :, 0]).astype(np.uint8) * 255
        diff_rg_mask = ((bgra[:, :, 2] - bgra[:, :, 1]) > 15).astype(np.uint8) * 255
        bgra_res_mask1 = cv2.bitwise_and(lower_mask, rg_mask)
        bgra_res_mask2 = cv2.bitwise_and(rb_mask, diff_rg_mask)
        bgra_mask = cv2.bitwise_and(bgra_res_mask1, bgra_res_mask2)

        # skin color mask in hsv
        hsv = cv2.cvtColor(np_batch[i,:,:,:], cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, (0, 0.23 * 255, 0), (50, 0.68 * 255, 255))

        # skin color mask in ycrcb
        ycrcb = cv2.cvtColor(np_batch[i,:,:,:], cv2.COLOR_BGR2YCrCb)
        lower_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 255, 255))
        cr_mask1 = (ycrcb[:, :, 1] <= (1.5862 * ycrcb[:, :, 2] + 20)).astype(np.uint8) * 255
        cr_mask2 = (ycrcb[:, :, 1] >= (0.3448 * ycrcb[:, :, 2] + 76.2069)).astype(np.uint8) * 255
        cr_mask3 = (ycrcb[:, :, 1] >= (-4.5652 * ycrcb[:, :, 2] + 234.5652)).astype(np.uint8) * 255
        cr_mask4 = (ycrcb[:, :, 1] <= (-1.15 * ycrcb[:, :, 2] + 301.75)).astype(np.uint8) * 255
        cr_mask5 = (ycrcb[:, :, 1] <= (-2.2857 * ycrcb[:, :, 2] + 432.85)).astype(np.uint8) * 255
        ycrcb_res_mask1 = cv2.bitwise_and(lower_mask, cr_mask1)
        ycrcb_res_mask2 = cv2.bitwise_and(cr_mask2, cr_mask3)
        ycrcb_res_mask3 = cv2.bitwise_and(cr_mask4, cr_mask5)
        ycrcb_res_mask12 = cv2.bitwise_and(ycrcb_res_mask1, ycrcb_res_mask2)
        ycrcb_mask = cv2.bitwise_and(ycrcb_res_mask12, ycrcb_res_mask3)

        # final skin mask
        bgra_hsv_mask = cv2.bitwise_and(bgra_mask, hsv_mask)
        bgra_ycrcb_mask = cv2.bitwise_and(bgra_mask, ycrcb_mask)
        final_mask = cv2.bitwise_or(bgra_hsv_mask, bgra_ycrcb_mask)
        final_mask = cv2.medianBlur(final_mask, 3)
        mask[i,:,:,0] = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        # maximal contour
        contours, _ = cv2.findContours(mask[i,:,:,:], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_idx = -1
        max_area = -1
        if len(contours) > 0:
            for j in range(len(contours)):
                contour = contours[j]
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_idx = j
            max_contour = contours[max_idx]
            cv2.drawContours(contour_img[i,:,:,:], [max_contour], 0, (0, 255, 0), 2)

        # resize
        resize = transforms.Resize((28, 28))

    return nd.concat(resize(nd.array(contour_img)).flatten(), resize(nd.array(mask)).flatten(), dim=1)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mask_contour", dest='mask', action='store_true',
                        help="Segmentation Mask and Contour Image are additionally used for training.")
    parser.add_argument("--no_mask_contour", dest='mask', action='store_false',
                        help="No use of Segmentation Mask and Contour Image")
    parser.set_defaults(mask=False)
    args = parser.parse_args()

    # hyperparameters
    lr, num_epochs, ctx = 1e-3, 200, try_gpu()
    batch_size = 128

    # load data
    augs = mx.image.CreateAugmenter(data_shape=(3, 224, 224),
                                    rand_crop=0.5,
                                    rand_mirror=True,
                                    brightness = 0.5,
                                    contrast = 0.5,
                                    saturation = 0.5,
                                    pca_noise = 0.2,
                                    inter_method = 10)

    def train_transform(data, label):
        data = data.astype(np.float32)
        for aug in augs:
            data = aug(data)
        return data, label

    def valid_transform(data, label):
        transformer = transforms.Compose([transforms.Resize((224, 224))])
        return transformer(data.astype(np.float32)), label

    trainset = data.vision.datasets.ImageFolderDataset("./dataset/valid_train", 1, train_transform)
    validset = data.vision.datasets.ImageFolderDataset("./dataset/valid_test", 1, valid_transform)

    trainloader = data.DataLoader(trainset, batch_size, True, num_workers=8, pin_memory=True)
    validloader = data.DataLoader(validset, batch_size, False)


    # model
    mobilenet = vision.mobilenet0_5(pretrained=True, ctx=ctx).features
    clf = nn.Sequential()
    with clf.name_scope():
        clf.add(nn.Dense(4096, activation='relu'),
                nn.Dense(4096, activation='relu'),
                nn.Dense(1024, activation='relu'),
                nn.Dense(6))

    clf.collect_params().initialize(init=init.Xavier(), ctx=ctx)

    # scheduler + trainer
    if args.mask:
        steps_epochs = [150, 175]
    else:
        steps_epochs = [150]
    it_per_epoch = math.ceil(1243 / batch_size)
    steps_iterations = [s * it_per_epoch for s in steps_epochs]
    schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.5)
    optimizer = mx.optimizer.RMSProp(learning_rate=lr, lr_scheduler=schedule)
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer_net = gluon.Trainer(params=mobilenet.collect_params(), optimizer=optimizer)
    trainer_clf = gluon.Trainer(params=clf.collect_params(), optimizer=optimizer)

    # training
    print("Use segmentation mask and contour image:", args.mask)
    loss_hist = train(mobilenet, clf, trainloader, validloader, criterion, trainer_net, trainer_clf, ctx,
                          batch_size, num_epochs, args.mask)



