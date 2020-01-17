# Handsign Classification

## Requirements
> pip install -r requirements.txt

## Dataset
For training, we used the following [Handsign Digit Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset). 

## Hand Segmentation

### Motivation
Our dataset is quite different to the validation dataset. The two main differences being:
> 1) Orientation of the hand  
> 2) Background and exposure

#### Problem 1
This can be easily solved with data augmentation by applying random rotations.

#### Problem2 
We want to solve this problem by hand segmentation. Once we separate the hand from the background and train on the segmentation masks the train and validation set should be very similar to each other.

### Theory and Approaches

#### First Approach
We trained a [Unet](https://arxiv.org/pdf/1505.04597.pdf) model on the [GTEA-gaze dataset](http://www.cbi.gatech.edu/fpv/) to segmentate hands. It did well on [GTEA-gaze](http://www.cbi.gatech.edu/fpv/), but failed on both our train and validation sets. The segmentation masks kind of resemble hands, but not really. 

#### Second Approach 
We follow the approaches of [Chai and Ngan, 1999](https://pdfs.semanticscholar.org/8055/dcb4e491dfe35b72e7817ce51e12184ce608.pdf) and [Basilio, 2011](https://pdfs.semanticscholar.org/462b/3024f2aeb547f9181e9466776461a353c5e9.pdf). The main idea is to transform the RGB images to the YCbCr color space. Since Y describes the luminance of the image, it can be ignored for the skin segmentation task (luminance does not affect the result of this method). Only Cb and Cr will be used for skin segmentation by defining the color range that human skin can have. Through extensive analysis, the authors came to the following definitions respectively 77<=Cb<=127, 133<=Cr<=173 and 80<=Cb<=120, 133<=Cr<=173. Both approaches worked quite well on the trainset. On the validation set, some results were ok, some were quite bad and some failed completely. Since the hand segmentation quality varies a lot on the validation set, we searched for further improvements (see approach 3 and 4). 

#### Third Approach
This follows the idea of the second approach, but additionally takes into account the HSV color space as proposed by [Rahman et al., 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7005726&tag=1). The color ranges are given by 0<=H<=25, 335<=H<=360, 0.2<=S<=0.6 and V>=40. The range for Cb and Cr are the same as proposed by [Chai and Ngan, 1999](https://pdfs.semanticscholar.org/8055/dcb4e491dfe35b72e7817ce51e12184ce608.pdf). The results are similar to the second approach with slight improvements.

#### Fourth Approach
This follows the proposal of [Kolkur et al., 2017](https://arxiv.org/pdf/1708.02694.pdf). It takes into consideration three color spaces RGBA, YCrCb and HSV. The selection rule for what color ranges human skin should have, is quite convoluted and best to be read in the linked paper. It includes over 15 ranges and relations between the values in the three color space, which allows the authors to achieve remarkable state-of-the-art results.  Our final code implements this method. The results on the validation set are however mixed and not very satisfying. However, it performs really good on the training set. From all four approaches, this one performs the best.

### Usage
If you have your own testset (named *test*), copy it into the folder *dataset* with the following structure: 
> *./dataset/test/class_labels/images_of_class.jpg*. 

If your dataset is not sorted into class folders, you can put all images into a dummy folder: 
>*./dataset/test/dummy_folder/all_images.jpg*.

Run the python script to perform hand segmentation:
> python hand_segmentation.py

<br/>

### Reasons for poor performance on validation set
> 1) The background could include objects with colors laying in the color range of human skin. We observed that some background pixels are segmentated as hand.

> 2) The images are not all taken in natural lighting, but rather where augmented artificially by hand. This will mess up the color relation between the various values in the 3 color space which human skin should have for natural images according to cited research. This explains why some segmentation masks do not even classify hand pixels as hand. Some even classify all pixels of the image as non-hand.

Have a look at *./dataset/train_mask* and *./dataset/valid_mask* to get an idea.

## Adaptive Gamma Correction
We tried two adaptive gamma correction approaches, namely [IAGCWD](https://arxiv.org/pdf/1709.04427.pdf) and [AGCWD](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0138-1). Both methods did not improved hand segmentation.

### Usage
Install requirements:
> pip install -r BEGAN_requirements.txt

Run the [code](https://github.com/carpedm20/BEGAN-tensorflow) with appropiate learning rates: 
> ./began *learning_rate_discriminator* *learning_rate_generator*


## Classification

### Data Augmentation
Data augmentation is applied to all the following approaches. In particular, we used custom implemented random rotation (because the validation set contains hand in all directions, whereas our training set only contains upright hands. This augmentation is most essential), random resized crop (always works well from my experience), random horizontal flip (a flip image would still makes sense and increase diversity, so why not?), jittering of brightness, contrast, saturation (because the validation set already contains instances of the same images with jittering of those parameters, in this case jittering the training data would make sense) and adding PCA noise (not sure if this helps, but probably would not hurt. That is why pca noise as an data augmentation is kept low). 

### Hand Segmentation Approach
Train classification on the obtained segmentation masks. This does not work well, since the segmentation masks for the trainset and validation set are very different to each other due to above described reasons.

### Pretrained Mobilenet + Finetuning
This approach is self-explanatory. We train a 3-layer fully-connected classifier on top of the pretrained [mobilenet](https://arxiv.org/pdf/1801.04381) with fixed weights and than fine-tune the whole model. 

#### Usage
> python train.py

### Pretrained Mobilenet + Segmentation Mask + Contour Image + Finetuning
Similar to the previous approach. We forward the input image through the pretrained [mobilenet](https://arxiv.org/pdf/1801.04381) to obtain the image features. Than we concatenate the image features, the segmentation mask (obtain by our hand segmentation algorithm described above) and the contour of the segmentation mask and feed it to the classifier. We subsequently train the classifier and than fine-tune the whole model. 

#### Usage
> python train.py --use\_mask\_contour

<br/>

### Observations and Further Though
Training on the trainset and evaluating on the validation set using the additional segmentation mask and contour image yields the highest validation accuracy of all approaches. However, the difference between images and segmentation masks in the training set and the validation set really manifested itself within the train and validation accuracies (mainly that the training images did not have a background or rather the background was always white). With segmentation mask and contour image, we achieved 100% accuracy on the trainset, but only 35% accuracy on the validation set. This result is kind of expected, since the images from training and validation set differ so much. 

The solution to negate those differences was hand segmentation, once the hand could be segmentated successfully the varying backgrounds do not matter anymore. However, two days before the deadline I had to realized that the hand segmentation problem could not been solved within the deadline time. With this in mind other approaches were harvested, but an idea also needs to be implemented, debugged and trained, which takes time and narrows down the number of possible approaches that can be realized within that time frame. Given the circumstances and the estimation that the hidden testset is probably more similar to the validation set than to the training set, we decided to directly train on the validation set and hope that the model does not overfit too much on this small amount of data (it certainly will, but we hope that at least it learns something and not just memorize the data). It is unorthodox and should not be done, but given the circumstances and the goal to achieve maximum accuracy on the black-box dataset, this seems like the most viable approach. Training and testing on the validation set (with a random 80/20 split), the [mobilenet](https://arxiv.org/pdf/1801.04381) and finetuning approach without segmentation mask and contour image performs the best. This makes sense, since the segmentation masks on the validation set were not that great in the first place. 

In hindsight, the best method to obtain the highest accuracy would have been to collect similar data. Since I did not find data similar enough to the validation set, I would have just taken a lot of pictures similar to the validation set. Take 10 pictures a minute for 10 hours a day for 10 days and you get 60000 images. Use the remaining 4 days to label the data and implement the code. Probably more effective, but at the same time also the best way to waste 2 weeks without learning much. So not much to regret here. If at least I would have found similar unlabelled data, than semi-supervised learning would have been an option. Appropiate self-supervised pre-task could have been [rotation, Gidaris et al., 2018](https://arxiv.org/pdf/1803.07728.pdf), colorization such as proposed by [Larsson et al., 2016](https://arxiv.org/pdf/1603.06668.pdf) and [Zhang et al., 2016](https://arxiv.org/pdf/1603.08511.pdf) or predicting relative position of image patches as proposed by [Doersch et al., 2016](https://arxiv.org/pdf/1505.05192.pdf) and [Noroozi et al, 2016](https://arxiv.org/pdf/1603.09246.pdf).

## Testing Usage
First name your folder containing the testset as *test*. The folder *test* should have each image sorted into its corresponding class folder named after their label (0-5). Have a look at *./dataset/train* and *./dataset/valid* to get an idea. Then run:
> python test.py (preferred, since the hand segmentation algorithm does not work well on articially augmented data)
or 
> python test.py --use\_mask\_contour 

## Backup
Running train.py will overwrite pretrained models and plots. If this happens by accident and you want to retrieve the files, go the the folder *Backup* and copy the files to root again. Note: test.py needs the saved models to be in root.

## Contact
Leo.Nguyen@gmx.de

## License
MIT License





