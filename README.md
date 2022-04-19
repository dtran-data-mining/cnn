# data-mining--cnn

**Assignment 3 - Convolutional Neural Networks (CNNs)**

This assignment focuses on the implementation of a convolutional neural network (CNN) using [PyTorch](https://pytorch.org/) and classification of data from the MNIST dataset, which consists of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

## Usage

Open the `run.sh` shell script and modify it according to the instructions below. *Please note: this project is currently only supported for macOS and Linux systems. If you wish to run it on Windows, you'll need to set it up accordingly.*


### Virtual Environment Setup

> Note: a virtual environment is not necessary to run the program, but it's highly recommended and is good practice for most Python projects, especially ML projects with many package dependencies such as this one.

Provided within the script are two commands to initialize a virtual environment. Make sure you have [anaconda](https://www.anaconda.com/) installed on your machine. You can also use other virtual environment packages, but PyTorch has an installation option directly for anaconda, so it's highly recommended. 

```shell
  source /Users/dtran/opt/anaconda3/etc/profile.d/conda.sh
  conda activate hw-env
```

1. Create a new virtual environment and call it `hw-env`. You can do this by running the command: 
  
   ```shell
   conda create hw-env
   ```

   You could also name it whatever you want, but make sure to modify the shell script to reflect its name.

2. Install the required [**package dependencies**](https://github.com/dtran421/data-mining--cnn#package-dependencies) for the program.
  
3. Change the path of `conda.sh` to the appropriate path for your machine. It should be located within the directory where anaconda was installed.

   ```shell
   source <path>/conda.sh
   ```


### Setting the Options

The main script of the program has a large array of command-line options that can be tweaked to your liking in order to influence the composition and training process of the CNN model. Within the shell script, you can make any adjustments to the program options as you desire. To learn more about the options, please see the [**options**](https://github.com/dtran421/data-mining--cnn#program-options) section.

> The most important option to note is the `mode` option, which is set to *test* by default. If you want to train the model from scratch, then set the option to *train*.

Below is the provided configuration for the model and training:

```shell
python3 main.py -mode=test -num_epoches=50 -ckp_path=checkpoint -learning_rate=0.005 -decay=0.075 -dropout=0.5 -p_huris=0.8 -batch_size=100 -channel_out1=64 channel_out2=64 -rotation=15 -fc_hidden1=500 -fc_hidden2=100 -MC=20
```


### Running the Program

With that done, now you're ready to start the program. Navigate to the program directory and run the following command to start the program.

```shell
  sh ./run.sh
```


## Program Options

The default settings for the options are specified within the `main.py` file of the program.

| Option        | Description                       | Type   | Default      |
|:-------------:|-----------------------------------|:------:|:------------:|
| mode          | train or test                     | string | *train*      |
| num_epoches   | num of epoches                    |   int  | *40*         |
| ckp_path      | path of checkpoint                | string | *checkpoint* |
| learning_rate | learning rate per epoch           | float  | *0.01*       |
| decay         | decay rate of learning rate       | float  | *0.1*        |
| dropout       | dropout probability               | float  | *0.5*        |
| batch_size    | batch size                        | int    | *100*        |
| channel_out1  | num of channels for conv layer 1  |   int  | *64*         |
| channel_out2  | num of channels for conv layer 2  | int    | *64*         |
| rotation      | transform random rotation         | int    | *10*         |
| fc_hidden1    | dim of neurons for linear layer 1 |   int  | *100*        |
| fc_hidden2    | dim of neurons for linear layer 1 | int    | *100*        |


## Additional Resources

- [Training a CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch CNN with MNIST](https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118)
- [CNN Layer Output Channels](https://discuss.pytorch.org/t/runtimeerror-mat1-and-mat2-shapes-cannot-be-multiplied-64x13056-and-153600x2048/101315)


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
* [tensorboard](https://pypi.org/project/tensorboard/)
