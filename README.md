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

* Python 3.5.2
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
                              [-d DATA_DIR] -r TF_REC -a {TFCNN,SimpleANN} -l
                              ALIAS [-s MODEL_DIR] [-e MODEL_EPOCH]

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
  -a {TFCNN,SimpleANN}  Model Architecture to Use
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

When using the 'prep' modes, you have to specify the location of data (via -d) which you will format into TFRecords.  
When using 'non-prep' modes, you have to specify location of TFRecords (via -r).

### Training

First of all, you have to gather and organize your image data!  
This CLI expects the following directory structure for DATA_DIR (location of images):

```
DATA_DIR/<class0>/image1.png
DATA_DIR/<class0>/image2.png
...
DATA_DIR/<class0>/imageN.png

DATA_DIR/<class1>/image1.png
DATA_DIR/<class1>/image2.png
...
DATA_DIR/<class1>/imageN.png

...
...

DATA_DIR/<classN>/image1.png
DATA_DIR/<classN>/image2.png
...
DATA_DIR/<classN>/imageN.png
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
-a SimpleANN \
-l cifar
```

The above command will do the following:
1. Collect all images in data/cifar/train along with their corresponding labels  
and store them in TFRecords file format in records/cifar_train_img.tfrecords
2. Train image model with "SimpleANN" architecture (see arch/arch.py) using data in TFRecords file  
    - the number of Epoch, Processed Images, and Loss value are printed
3. Save trained model in following path:  
*models/SimpleANN/\<datetime\>\_SimpleANN\_cifar\_\<epoch\>.mdl.data*
    - the number of saved models may be varied in **common/constants.py**

**To re-run same training without TFRecords preparation, use:**
```
python3 imageclassifier_cli.py \
-m trn \
-r records/cifar_train_img.tfrecords \
-a SimpleANN \
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
-a SimpleANN \
-l cifar
```

The above command will do the following:
1. Collect all images in data/cifar/test along with their corresponding labels  
and store them in TFRecords file format in records/cifar_test_img.tfrecords
2. Load pre-trained image model with latest epoch in following path:  
*models/SimpleANN/\<datetime\>\_SimpleANN\_cifar\_\<epoch\>.mdl.data*
3. Classify data in TFRecords file using loaded model
    - number of processed images is printed during execution
4. Print (and save to log file) the accuracy

**To re-run same testing without TFRecords preparation, use:**
```
python3 imageclassifier_cli.py \
-m tst \
-r records/cifar_test_img.tfrecords \
-a SimpleANN \
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

### Adding architectures

Other network architectures may be added through **arch/arch.py**.  
Make sure to update **model_arch** in image_classifier.py to add it as an architecture option in CLI.

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
