import os
import glob
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from tqdm import tqdm
from pathlib import Path
from skimage.transform import resize
from sklearn.utils import Bunch
from joblib import Parallel, delayed





def load_image_files(container_path, labels_path, dimension=(64, 64)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    print("cououcou")
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir()]
    #print(folders)
    #exit()
    #labels = pd.read_csv(labels_path, encoding='utf8', engine='python')
    #print(np.array(labels))
    #exit()
    #categories = [fo.name for fo in folders]
    #label
    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    #for i, direc in enumerate(folders):
    #    print(i)

    #for file in tqdm(direc.iterdir()):
    for file in tqdm(folders):
        img = mpimg.imread(file)
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        images.append(img_resized)
        #target.append(i)
    flat_data = np.array(flat_data)
    #target = np.array(target)
    images = np.array(images)

    bunch = Bunch(data=flat_data,
                 #labels=labels,
                 images=images,
                 DESCR=descr)
    

    return bunch

#results = Parallel(n_jobs=-1)(load_image_files("../data/train_images/", "../data/train.csv"))
#exit()
#results = Parallel(n_jobs=-1, verbose=1, backend="threading")(map(delayed(load_image_files),"../data/test_images/", "../data/train.csv"))

#image_dataset = load_image_files("../data/train_images/", "../data/train.csv")
#pickle.dump(image_dataset, open("train.p", "wb", -1))
#print(image_dataset)




from joblib import Parallel, delayed
import multiprocessing


image_dir = Path("../data/train_images/")
#image_dir = glob.glob("../data/train_images/*")
folders = [directory for directory in image_dir.iterdir()]
#folders = [directory for directory in image_dir]
#print(folders)
#exit()
#print([(i, f) for i in enumerate(folders)])
#exit()
labels_path = "../data/train.csv"
labels = pd.read_csv(labels_path, encoding='utf8', engine='python')
flat_data = []
images = []

def processInput(file):
    img = mpimg.imread(file)
    img_resized = resize(img, (64, 64), anti_aliasing=True, mode='reflect')
    flat_data.append(img_resized.flatten()) 
    images.append(img_resized)
    bunch = Bunch(data=flat_data,
                 labels=labels,
                 images=images)

    return bunch

#results = Parallel(n_jobs=-1)(delayed(processInput)(file) for file in tqdm(folders))
#print(results)
#print("Ecriture du pickle")
#pickle.dump(results, open("train3.p", "wb", -1))
#print("Fin")

with open("./train2.p", mode='rb') as f:
    train = pickle.load(f)

print(train)
