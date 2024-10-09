Code of P-AT with mnist dataset
===
You can utilize train_mnist.py to setup a new training process with mnist dataset. For example:<br>
```
python train_mnist.py  --hyperparameters XX
```
There are two models that we have been tested: ToyNet (A simple CNN structure termed mnist_net) and ResNet (please modify the stem layer as nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False), remove the first maxpool layer--- just as the process for CIFAR, and change the number of the output layer to 11)<br>

In the content of train_mnist.py, LabelSmoothingLoss is the trap smoothing loss and makeRandom is the padding generative function, which are the significant components of P-AT.<br>
You can change the code of ```P_X = X[:10, :, :, :]``` to control the percentage of padding data in each epoch.

After AT, you can use eval_mnist.py to test the performance of your models.
We provide the weights (.pth) file of toyNet, you can check out the performance of P-AT in Mnist dataset.
Concretely, 1.pth and 4.pth are the weights of Standard-AT(FGSM) and P-AT(Fgsm) with toyNet<br>

As the weight files are larger than 25MB, if you are interesting in relative weight files of ResNet or the weight with CIFAR and MINI-ImageNet, please contact us. 

