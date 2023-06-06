import torch
import numpy as np
import matplotlib.pylab as plt
from resonet.utils.eval_model import load_model
from sklearn import metrics 
from resonet.loaders import H5SimDataDset

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter as arg_formatter

parser = ArgumentParser(formatter_class=arg_formatter)
parser.add_argument("input", type=str, help="input training data h5 file")
parser.add_argument("--trainRange", type=int, nargs=2, default=None)
parser.add_argument("model", type=str, help="direct path to the model file should end in .nn")
parser.add_argument("--arch", type=str, choices=["le", "res18", "res50", "res34", "res101", "res152"],
                        default="res34", help="architecture selector")
parser.add_argument("outdir", type=str, help="store output files here (will create if necessary)")
parser.add_argument("--gpu",action='store_true')
args = parser.parse_args()

master_file = args.input
train_start = args.trainRange[0]
train_stop = args.trainRange[1]
model = args.model
arch = args.arch
output = args.outdir



label_sel = ['is_multi']
half_precision = False
h5imgs = 'images'
h5label = 'labels'
dev = 'cpu' 
if args.gpu:
    dev = 'cuda:0' # optional if user has access to gpu, recommended
use_geom = False
transform = False

common_args = {"dev":dev,"labels": h5label, "images": h5imgs,
                   "use_geom": use_geom, "label_sel": label_sel,
                   "half_precision": half_precision}
# Grabs the images that need to be predicted by the model, in this case the same images used to train
train_imgs = H5SimDataDset(master_file, start=train_start, stop=train_stop ,transform=transform, **common_args)

# load the model, first arg is the model .nn file, second arg is arch (see the train.log)
model = load_model(model,arch) 
model = model.to(dev)

# declare the size 
n = train_stop - train_start
all_labels = np.zeros(n)
all_pred = np.zeros(n)

# use loop to grab the labels ths is our label truth
for i in range(len(train_imgs)):
    print(i)
    image,label = train_imgs[i]
    image = image[None]
    # img should be 1x1x512x512 tensor 
    raw_pred = torch.sigmoid(model(image))
    is_multi = int(torch.round(raw_pred).item())
# Grabs all labels from the orignal images, and the predictions 
   
   # give theee index 
    all_labels[i] = label
    all_pred[i] = is_multi

# Create the confusion matrix
confusion_matrix = metrics.confusion_matrix(all_labels, all_pred)
print(confusion_matrix)

# Adding features to the Confusion Matrix
fig , ax = plt.subplots(figsize=(10, 10))
fig.set_size_inches((3,3))
fs = 10
ax.matshow(confusion_matrix, cmap=plt.cm.Pastel1)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(x=i, y=j, s=confusion_matrix[j, i], va='center', ha='center', fontsize = fs)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['True','False'], fontsize = fs)
        ax.set_yticklabels(['True','False'], fontsize = fs)
        ax.set_title('Predicted label', fontsize = fs)
        plt.ylabel('True label', fontsize = fs)
        plt.subplots_adjust(left=.2,right=.99, top=.83, bottom=.07)
        plt.savefig(output,dpi=250)
        plt.show()


