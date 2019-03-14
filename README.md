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

### Project Files
 - The files `test_vgg16.py`, `test_vgg19.py` are simple examples, that can be executed only by cloning the project.
 - The file `test_vgg19_trainable.py` is a simple example of test and training of a CNN model. Also can be executed only by cloning the project
 - The file `test_vgg19_all.py` is a complete classification model that require other files and a dataset more elaborate.

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
