# Tensorflow VGG Transfer Learning - Trainable

This is a Tensorflow implemention of VGG 16 and VGG 19 based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg), developed for [Tensor r1.0](https://www.tensorflow.org/)

We have modified the implementation of <a href="https://github.com/machrisaa/tensorflow-vgg">tensorflow-vgg</a> so that the model support different image formats (jpg,png,jpeg). Additionally we have developed a function 
In addition we have developed a function to perform model training using mini-batches and epochs. The project own us modify the original model and load the weights pre-trained in the required layer.

### Download weights pre-trained:
The guiding project (<a href="https://github.com/machrisaa/tensorflow-vgg">tensorflow-vgg</a>) for the development of this repository uses transfer learning with the weights obtained from the training of the VGG network with dataset [Imaginet](http://image-net.org/challenges/LSVRC/2016/index).

  -__OPTION A:__ To use the VGG networks, the npy files for [VGG16 NPY](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) or [VGG19 NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs) has to be downloaded. Uploaded in Mega.nz repositories.

  -__OPTION B:__ You can download the pre-trained weights by terminal. You should install [Megatools](https://megatools.megous.com/) and follow the guide in the website <a href="http://luiszambrana.com.ar/bajar-archivos-de-mega-por-terminal">[link]</a>.
  
  -__OPTION C:__ You can also download the files from a repository in dropbox, by terminal:
  ```
              vgg16.npy[528MB]: wget https://www.dropbox.com/s/8a8rei66f72um4i/vgg16.npy
              vgg19.npy[548MB]: wget https://www.dropbox.com/s/691wtp4oq5ip38p/vgg19.npy
  ```

### Content
  - ./data: This folder contains samples of images to execute the examples of the folder ./scripts/tester/.
  - ./notebooks: This folder contains files with .ipynb extension for Jupyter .
  - ./script: This folder contains the scripts neccesary to execute the convolutional networks.
    - ./scripts/nets: This folder contains the classes responsible for building a tensorflow graph and provides functions for manipulating the convolutional network model.
    - ./scripts/tools: This folder contains auxiliary functions to execute the models.
    - ./scripts/tester: This folder contains an example for each convolutional network class.
  - ./weights: You should create this folder and add the downloaded npy files.
  - ./graphs: If you want to use Tensorboard, it is recommended to create this folder to store the necessary files that will run the local server with Tensorboard

### Project Files
 - The files `t_vgg19.py`, `tester_vgg19_trainable.py` are simple examples, that can be executed only by cloning the project.
 - The file `t_vgg19_trainable.py` is a simple example of test and training of a CNN model. Also can be executed only by cloning the project
 - The file `t_vgg19_batch.py` is a complete classification model that require other files and a dataset more elaborate.

### Usage
Use this to build the VGG object
```
vgg = vgg19.Vgg19()
vgg.build(images)
```
or
```
vgg = vgg16.Vgg16()
vgg.build(images)
```
or by training
```
vgg = vgg19.Vgg19([path_vgg.npy], load_weight_fc)
vgg.build(images, train_mode)
```
The `[path_vgg.npy]` is the path where the file '.npy' is located.
The `load_weight_fc` is of type bool, this variable own us reset the weights of the fully-connected layers, but can train the model with a new dataset.
The `images` is a tensor with shape `[None, 224, 224, 3]`. 
The `train_mode` is of the type bool, this variable own us enable or disable the dropout layer.

>Trick: the tensor can be a placeholder, a variable or even a constant.

### Access to parameters
All the VGG layers (tensors) can then be accessed using the vgg object. For example, `vgg.conv1_1`, `vgg.conv1_2`, `vgg.pool5`, `vgg.prob`,etc. Even recover the weights of the model layers `vgg.var_dict[(name, idx)]`, where the variable `name` is the name assigned to a layer and `idx` indicates the type of weights we want to recover, filters/weigths or biases.

The `test_vgg16.py` and `test_vgg19.py` contain the sample usage.

### Load Dataset
The class for loading a dataset and generating sub sets for the minibatchs is in the file 'dataseTools.py', this class requires three input data:
  - *data/[image].jpg:* The directory that contains the input images.
  - *data-train.csv:* The file containing the list of all training images. This contains two main fields [ 'image name (string)','class (string)','tag or class (int)'].
  - *minibatch:* This variable stores the size of the minibatch.

### Enverioment GPU
Google Colaboratory: <a href='https://drive.google.com/drive/folders/1FQIrTWLObCfFOjqQojzx73qQ2g2X6g-f?usp=sharing'>Colab Workspace<a/>

### Additional
  - Basic course of Machine Learning and Tensorflow: <a href='https://paper.dropbox.com/doc/MACHINE-LEARNING-ML--AZTcNGc3Q~wJa~hst1IjrRguAQ-Wakz7vrDO4AmkK9zhGO9e'>MACHINE-LEARNING-ML</a>
  - Basic documentation of Tensorflow: <a href='https://paper.dropbox.com/doc/TENSORFLOW--AZSCmtaqm6hHuzONzQ1DQYsiAQ-Tt6UfeyV3zEm7ds2ZeO5X'>TENSORFLOW</a>
  - Tensorflow practical course: <a href='https://drive.google.com/drive/folders/0B41Zbb4c8HVyMHlSQlVFWWphNXc?usp=sharing'>Drive Folder</a>
