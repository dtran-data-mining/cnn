# data-mining--cnn

**Assignment 2 - Classification and Clustering**

This assignment focuses on the implementation of a convolutional neural network (CNN) using PyTorch and classification of data from the MNIST dataset, which consists of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

## Simple Setup

Open the `run.sh` shell script and modify it according to the instructions below.


### Virtual Environment Setup

> Note: a virtual environment is not necessary to run the program, but it's highly recommended and is good practice for most Python projects, especially ML projects with many package dependencies such as this one.

Provided within the script are two commands to initialize a virtual environment. Make sure you have [**anaconda**](https://www.anaconda.com/) installed on your machine. You can also use other virtual environment packages, but PyTorch has an installation option directly for anaconda, so it's highly recommended. 

```shell
  source /Users/dtran/opt/anaconda3/etc/profile.d/conda.sh
  conda activate hw-env
```

1. Create a new virtual environment and call it `hw-env`. You can do this by running the command: 
  
   ```shell
   conda create hw-env
   ```

   You can also name it whatever you want, but make sure to modify the shell script to reflect its name.

2. Install the required package dependencies for the program using the following command:

   ```shell
   conda install --file requirements.txt
   ```
  
3. If you're on **Windows**, then you can remove the following command from the script (the first command within the file). If you're on **macOS**, then you need to change the path of `conda.sh` to the appropriate path for your machine. It should be located within the directory where anaconda was installed.

   ```shell
   source <path>/conda.sh
   ```


### Setting the Options

The main script of the program has a large array of command-line options that can be tweaked to your liking in order to influence the composition and training process of the CNN model. Within the shell script, you can make any adjustments to the program options as you desire. To learn more about the options, please see the [**options**](https://github.com/dtran421/data-mining--cnn/edit/master/README.md#program-options) section.

> The most important part to note is the `mode` option, which is set to *test* by default. If you want to train the model from scratch, then see the section on training.

Below is the default configuration for the model and training:

```shell
python3 main.py -mode=test -num_epoches=50 -ckp_path=checkpoint -learning_rate=0.005 -decay=0.075 -dropout=0.5 -p_huris=0.8 -batch_size=100 -channel_out1=64 channel_out2=64 -rotation=15 -fc_hidden1=500 -fc_hidden2=100 -MC=20
```


### Running the Program

With that done, now you're ready to start the program. Navigate to the program directory and run either of the following commands to start the program.

**Windows:**
```shell
  sh ./run.sh
```

**macOS:**
```shell
  sh ./run.sh
```


## Advanced Setup

ASD

## Program Options

| Option        | Description   | Type   | Default |
|:-------------:|:-------------:|:------:|:-------:|
| *mode*        | ............. | string | test    |
| col 2 is      | centered      |   $12  |
| zebra stripes | are neat      |    $1  |


## Package Dependencies

The `model.py` class file makes use of the following packages:

* [PyTorch](https://pytorch.org/)
  * [torch](https://pytorch.org/docs/stable/torch.html)
  * [torch.nn](https://pytorch.org/docs/stable/nn.html)

The `main.py` script makes use of the following packages:

* [PyTorch](https://pytorch.org/)
  * [torch](https://pytorch.org/docs/stable/torch.html)
  * [torch.nn](https://pytorch.org/docs/stable/nn.html)
  * [torch.optim](https://pytorch.org/docs/stable/optim.html)
  * [torch.autograd](https://pytorch.org/docs/stable/autograd.html)
  * [torch.utils.data](https://pytorch.org/docs/stable/data.html)
  * [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)
* [torchvision](https://pytorch.org/vision/stable/index.html)
  * [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
  * [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)
