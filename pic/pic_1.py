#ls 相关实验绘图
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
# label_ls=np.linspace(start=11,stop=20,num=10,dtype=int)
label_ls=np.linspace(start=0,stop=1,num=11)
# label_ls[4]=0.34
plt.xticks(label_ls,fontsize=20)

plt.yticks(fontsize=20)

print(label_ls)



# untargeted
#clean
# before = np.array([99.03,27.54,1.03,0.7,0.4,0,0,0,0,0,0])
# after = np.array([99.03,27.54,1.03,0.7,0.4,0,0,0,0,0,0])

# fgsm
# before = np.array([98.33,97.70,94.56,82.37,1.04,0,0,0,0,0,0])
# after = np.array([98.33,97.70,94.56,82.37,1.04,0,0,0,0,0,0])
# # pgd
# before = np.array([98.99,98.32,96.03,93.24,1.04,0,0,0,0,0,0])
# after = np.array([98.99,98.32,96.03,93.24,1.04,0,0,0,0,0,0])

# # padding
before = np.array([96.99,95.32,93.24,82.96,20,0,0,0,0,0,0])
after = np.array([96.99,98.92,98.03,96.24,78,84,93,98,98,99,99])


# no_trap_smoothing_16.pth
# before = np.array([99.04,97.77,95.65,92.64,9.04,0,0,0,0,0,0])
# after = np.array([99.04,97.77,95.65,92.64,9.04,0,0,23,64,90,98])
lw=4
plt.plot(label_ls, before,"r-o",
         lw=lw, label='%s ' % ('Before detecting'),ms=12)
plt.plot(label_ls, after,"b--p",
         lw=lw, label='%s' % ('After detecting'),ms=12)
# plt.plot(label_ls, after_fgsm,"g--v",
#          lw=lw+2, label='%s ' % ('fgsm--after detecting'),ms=10)
# plt.plot(label_ls, before_pgd,"r-.o",
#          lw=lw, label='%s ' % ('pgd--before detecting'),ms=10)
# plt.plot(label_ls, after_pgd,"r-.v",
#          lw=lw+2, label='%s ' % ('pgd--after detecting'),ms=10)

# plt.plot(fpr_dict["micro"], tpr_dict["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_dict["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr_dict["macro"], tpr_dict["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc_dict["macro"]),
#          color='navy', linestyle=':', linewidth=4)


x_major_locator=MultipleLocator(0.1)
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# plt.xlim([11,20])
# plt.title("The Impact of Ls Rate(Untargeted attack)",fontsize=25)
plt.ylim([0.0, 105])
plt.xlim([0.0,1.05])
plt.yticks()
plt.xlabel('perturbation ε ', fontsize=25)
# plt.ylabel('Accuracy', fontsize=25)
plt.legend(loc="center right",fontsize=15)
plt.savefig("1_4.png",dpi=512,bbox_inches='tight')
plt.show()
