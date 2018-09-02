# Image Classifier CLI

An easy-to-use CLI tool for training and testing image classifiers.

## Key Features
* Can handle ANY image size (but you need to specify it!)
* Can handle ANY number of labels

## Limitations
* All data are assumed to be of SAME size
* Classes are based only on existing data

## Getting Started

### Prerequisites

* Python3
* Numpy
* TensorFlow

```
pip install tensorflow-gpu numpy
```

Note that depending on your environment (OS, GPU, etc.) the required TensorFlow version may vary.  
This CLI was developed and tested on a machine with following specs:

* OS: Ubuntu 16.04 64-bit
* GPU: GeForce 940MX, 2 GB RAM
* CUDA: v9.0

Of course, you may also choose to use TensorFlow CPU version :)

### Installing

```
git clone https://github.com/cjbayron/imageclassifier-cli.git
```

## Running this CLI

### Parameters

To see all the parameters, run the following on the base directory (imageclassifier-cli/):

```
python3 imageclassifier-cli.py -h
```

You should see the following:

```
usage: imageclassifier_cli.py [-h] -m {trn_prep,tst_prep,trn,tst}
                              [-d DATA_DIR] -r TF_REC -a
                              {ImageGRU,ImageLSTM,SimpleANN,TFCNN} -l ALIAS
                              [-s MODEL_DIR] [-e MODEL_EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -m {trn_prep,tst_prep,trn,tst}
                        Execution mode:
                            trn_prep - prepare data as TFRecords then execute training
                            tst_prep - prepare data as TFRecords then execute testing
                            trn - Execute training (no preparation)
                            tst - Execute testing (no preparation)
  -d DATA_DIR           Location of images to use
                         - used for trn_prep and tst_prep
  -r TF_REC             TFRecords file
                         - NAME of TFRecords file to be created (for trn_prep, tst_prep)
                         - PATH of TFRecords file to be used (for trn, tst)
  -a {ImageGRU,ImageLSTM,SimpleANN,TFCNN}
                        Model Architecture to Use
  -l ALIAS              Alias for trained model (e.g. name of data)
  -s MODEL_DIR          Location of Saved Model
                         - optional; used only for tst and tst_prep
  -e MODEL_EPOCH        Epoch (load model saved at end of specific epoch)
                         - optional; used only for tst and tst_prep
```

### Modes

1. *trn_prep* - Save image data as TFRecords, then perform training.
2. *tst_prep* - Save image data as TFRecords, then perform training.
3. *trn* - Perform training. Saves trained model in models/
4. *tst* - Perform testing. Loads trained model, classifies test data, and prints accuracy

When using the 'prep' modes, you have to specify the location of data (via **-d**) which you will format into TFRecords.  
When using 'non-prep' modes, you have to specify location of TFRecords (via **-r**).

### Training

First of all, you have to gather and organize your image data!  
This CLI expects the following directory structure for DATA_DIR (location of images):

```
DATA_DIR
│
├── <class0>
│   ├── image1.png
│   ├── image2.png
│   ...
│   └── imageN.png
│
├── <class1>
│   ├── image1.png
│   ├── image2.png
│   ...
│   └── imageN.png
...
│
└── <classN>
    ├── image1.png
    ├── image2.png
    ...
    └── imageN.png
```

As mentioned, this CLI assumes all these images to be of same size.  
Prior to training, make sure that **IMG_SHAPE** is set correctly in **common/constants.py**.  
Take note that **IMG_SHAPE** is also used by our image classifier  
to set the input layer size of the available architectures.

**Now, let's start training!**  
As an example, I will use the CIFAR-10 dataset in PNG format and store it in data/:
* [CIFAR-10 PNG](https://pjreddie.com/projects/cifar-10-dataset-mirror/)

and re-organize the images as follows:
```
data/cifar/train/airplane/10008_airplane.png
...
data/cifar/train/truck/9996_truck.png
```

**Sample execution:**
```
python3 imageclassifier_cli.py \
-m trn_prep \
-d data/cifar/train \
-r cifar_train_img \
-a TFCNN \
-l cifar
```

The above command will do the following:
1. Collect all images in *data/cifar/train* along with their corresponding labels  
and store them in TFRecords file format in *records/cifar_train_img.tfrecords*
2. Train image model with *TFCNN* architecture (see **arch/arch.py**) using data in TFRecords file  
    - the number of Epoch, Processed Images, and Loss value are printed
3. Save trained model in following path:  
*models/TFCNN/\<datetime\>\_TFCNN\_cifar\_\<epoch\>.mdl.data*
    - the number of saved models may be varied in **common/constants.py**

**Sample output:**
```
Found 50000 images. Creating TF records... 
Done.
Training on 50000 images for 5 epochs...
[2018-09-02 20:59:40.869] Epoch: 1/5, Processed: 0/50000, Loss: 3.386968
[2018-09-02 20:59:41.252] Epoch: 1/5, Processed: 3200/50000, Loss: 1.743668
[2018-09-02 20:59:41.638] Epoch: 1/5, Processed: 6400/50000, Loss: 2.058634
[2018-09-02 20:59:42.035] Epoch: 1/5, Processed: 9600/50000, Loss: 1.742383
[2018-09-02 20:59:42.428] Epoch: 1/5, Processed: 12800/50000, Loss: 1.573048
...

[2018-09-02 21:21:44.043] Epoch: 20/20, Processed: 35600/50000, Loss: 0.049832
[2018-09-02 21:21:45.492] Epoch: 20/20, Processed: 38800/50000, Loss: 0.039704
[2018-09-02 21:21:46.948] Epoch: 20/20, Processed: 42000/50000, Loss: 0.117836
[2018-09-02 21:21:48.396] Epoch: 20/20, Processed: 45200/50000, Loss: 0.006848
[2018-09-02 21:21:49.842] Epoch: 20/20, Processed: 48400/50000, Loss: 0.029094
Model saved in path: models/TFCNN/20180902_2121_TFCNN_cifar_20.mdl
```

**To re-run same training without TFRecords preparation, use:**
```
python3 imageclassifier_cli.py \
-m trn \
-r records/cifar_train_img.tfrecords \
-a TFCNN \
-l cifar
```

### Testing

Perform same initial steps done for Training (if you haven't done yet!).

**Sample execution:**
```
python3 imageclassifier_cli.py \
-m tst_prep \
-d data/cifar/test \
-r cifar_test_img \
-a TFCNN \
-l cifar
```

The above command will do the following:
1. Collect all images in *data/cifar/test* along with their corresponding labels  
and store them in TFRecords file format in *records/cifar_test_img.tfrecords*
2. Load pre-trained image model with latest epoch in following path:  
*models/TFCNN/\<datetime\>\_TFCNN\_cifar\_\<epoch\>.mdl.data*
3. Classify data in TFRecords file using loaded model
    - number of processed images is printed during execution
4. Print (and save to log file) the accuracy

**Sample output:**
```
Found 10000 images. Creating TF records... 
Done.
[2018-09-02 21:24:47.251] Starting classification. Using model in:
models/TFCNN/20180902_2121_TFCNN_cifar_20.mdl
[2018-09-02 21:24:49.584] Processed 100/10000 images
[2018-09-02 21:24:49.728] Processed 1100/10000 images
[2018-09-02 21:24:49.866] Processed 2100/10000 images
[2018-09-02 21:24:50.005] Processed 3100/10000 images
[2018-09-02 21:24:50.144] Processed 4100/10000 images
[2018-09-02 21:24:50.281] Processed 5100/10000 images
[2018-09-02 21:24:50.418] Processed 6100/10000 images
[2018-09-02 21:24:50.576] Processed 7100/10000 images
[2018-09-02 21:24:50.734] Processed 8100/10000 images
[2018-09-02 21:24:50.898] Processed 9100/10000 images
[2018-09-02 21:24:51.074] Accuracy: 75.320000
```

**To re-run same testing without TFRecords preparation, use:**
```
python3 imageclassifier_cli.py \
-m tst \
-r records/cifar_test_img.tfrecords \
-a TFCNN \
-l cifar
```

NOTE: If you have multiple trained models having SAME architecture, SAME alias and are located in SAME folder,  
the LATEST trained model is loaded. If this is not the intended behavior in such cases, then use alias (-l).

### ALIAS (-l)

The intended use of -l (alias) is to distinguish models having same architecture but are trained with different dataset or hyperparameters. For example, I can use aliases of "mnist" and "cifar" for training a  
SimpleANN network on MNIST and CIFAR-10 dataset respectively.  
This will leave me with two distinct models on same folder:

*models/SimpleANN/\<datetime\>\_SimpleANN\_mnist\_\<epoch\>.mdl.data*  
*models/SimpleANN/\<datetime\>\_SimpleANN\_cifar\_\<epoch\>.mdl.data*

Or if I want to train two different CIFAR models using different learning rates, I can use aliases to get the following:

*models/SimpleANN/\<datetime\>\_SimpleANN\_cifarlrn0.05\_\<epoch\>.mdl.data*  
*models/SimpleANN/\<datetime\>\_SimpleANN\_cifarlrn0.10\_\<epoch\>.mdl.data*

### MODEL_DIR (-s) and MODEL_EPOCH (-e)

-s *dir*
* You're basically telling the program to look for the pre-trained model in *dir*.  

-e *num*
* You're telling the program to use the model saved at epoch *num* for testing.  
(You may save the model at specific epoch intervals via **NUM_EPOCH_BEFORE_CHKPT** in **common/constants.py**)

### Architectures

Classes for neural network architectures are in **arch/arch.py**. Currently, the following are implemented:  
* [_TFCNN_](https://www.tensorflow.org/tutorials/estimators/cnn) - TensorFlow CNN Architecture
* _SimpleANN_ - Neural Net w/ Single Hidden Layer
* _ImageLSTM_ - Basic LSTM RNN for Images
* _ImageGRU_ - Basic GRU RNN for Images

Other network architectures may be added. Just make sure to update **MODEL_ARCH** in **image_classifier.py**  
to add them as architecture option in CLI.  

### Hyperparameters

You may modify hyperparameters in **common/constants.py** and **arch/arch.py**.

## Built With

* [TensorFlow](https://www.tensorflow.org/)

## Authors

* **Christopher John Bayron**
    * [Github](https://github.com/cjbayron)
    * [LinkedIn](https://www.linkedin.com/in/christopher-john-bayron)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
