Running train_fg_inil_pat.py to train models (ResNet,VGG,GoogleNet,MobileVit) with Fast-FGSM AT or PGD-AT. 

The parameters that should be considered in train_fg_inil_pat.py is trap smoothing factor, size of padding data and whether utilize trap smoothing loss at AEs generation stage.

The specific codes of trap smoothing loss and padding genreative function (makeRandom) are available at utils_cifar_out.py. 
