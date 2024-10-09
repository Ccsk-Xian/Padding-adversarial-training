You can utilize code like 
```
python trainpgdat.py  --model ResNet18 --arch_name ResNet18_1 --trial 1 --dataset miniImage --cuda 1
```
and 
```
python evalpgd.py --model ResNet18 --arch_name exp1_1 --dataset miniImage --weights /root/distill/save/teacher_model/ResNet18_miniImage_lr_0.05_decay_0.0005_trial_1/ResNet18_best.pth
```
to train and test the performance for the models with Mini-ImageNet.<br>

Notably, Apart from the normal hyper-parameters like batch_size, learning rate and so on, the performance of the final P-AT models is a joint result with the size of perturbation, trap_smoothing factor, percentage of padding data and the quality of padding generative function.<br>

According to our experimental result, we give the following guideline for researchers who interesting in P-AT:
1. The larger the value of the smoothing factor, the lower the original accuracy (slight impact) and the stronger the trap effect.
2. The adversarial perturbation of P-AT can be set larger than that of standard AT. For instance: 8/255 for standard AT and 16/255 for P-AT with the same basic attacks as former
3. The quality of padding generative function is very important. We strong call for researcher to study this issues. Notably, it shows distinct performance for the different size of models and datasets.
4. We notice that different batch_size have obvious impact for the models with different volumn.
5. We notice that the volumn of padding data is not important than we thought. Nevertheless, when the volumn of padding data is exceed of certain threshold, it would harm for the native accuracy and AEs detection performance.  
