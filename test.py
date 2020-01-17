import mxnet.gluon.data.vision.transforms as transforms
from mxnet import init
from mxnet.gluon import nn, data
from mxnet.gluon.model_zoo import vision
from train import *

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mask_contour", dest='mask', action='store_true',
                        help="Segmentation Mask and Contour Image are additionally used for training.")
    parser.add_argument("--no_mask_contour", dest='mask', action='store_false',
                        help="No use of Segmentation Mask and Contour Image")
    parser.set_defaults(mask=False)
    args = parser.parse_args()

    # hyperparameter
    ctx = try_gpu()
    batch_size = 128

    # load data
    def test_transform(data, label):
        transformer = transforms.Compose([transforms.Resize((224, 224))])
        return transformer(data.astype(np.float32)), label

    validset = data.vision.datasets.ImageFolderDataset("./dataset/test", 1, test_transform)
    testloader = data.DataLoader(validset, batch_size, False)

    # model definition
    mobilenet = vision.mobilenet0_5(pretrained=True, ctx=ctx).features
    clf = nn.Sequential()
    with clf.name_scope():
        clf.add(nn.Dense(4096, activation='relu'),
                nn.Dense(4096, activation='relu'),
                nn.Dense(1024, activation='relu'),
                nn.Dense(6))

    # load weights
    print("Use segmentation mask and contour image model:", args.mask)
    if args.mask:
        mobilenet.load_parameters("net_mask_contour.params", ctx=ctx)
        clf.load_parameters("clf_mask_contour.params", ctx=ctx)
    else:
        mobilenet.load_parameters("net.params", ctx=ctx)
        clf.load_parameters("clf.params", ctx=ctx)

    # evaluate
    test_acc = evaluate_accuracy(testloader, mobilenet, clf, ctx, args.mask)
    print("Test Accuracy:", test_acc)