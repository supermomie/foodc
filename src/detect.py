import os, glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.utils import Bunch
from skimage.io import imread
from skimage.transform import resize
from imgaug import augmenters as iaa
import six.moves.cPickle as pickle


data = pd.read_csv("../data/train.csv")
X_train , y_train = data['ImageId'], data['ClassName']

def samples(data, percent, seed=1):

    sample = data.sample(n=(round(percent*data.shape[0])), random_state=seed)
    return sample

sample = samples(data, 0.1)
print(sample)
num_of_samples = sample.groupby(['ClassName']).size().reset_index(name='Count')['Count']


print(len(num_of_samples))

print(num_of_samples.tolist())







def load_image_files(container_path, dimension=(64, 64)):
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
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    print(folders)
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        print(i)

        for file in direc.iterdir():
            img = mpimg.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    bunch = Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)
    

    return bunch


#image_dataset = load_image_files("../data/")
#pickle.dump(image_dataset, open("train.p", "wb", -1))
#print(image_dataset)

#TODO compresser l'image!!!
#exit()






training_file = './train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

print([t for t in train])
#print(train['images'])
exit()











def loadIMG(data):
    _all = []
    files = os.listdir(data)

    #d = [np.array(Image.open(v)) for v in data+'377ed8288b.jpg']
    #print(d)
    exit()
    #print(files)
    for num, x in enumerate(files):
        p = data+x
        img = IMG(p)
        _all.append(img)
    return _all


def IMG(image):
    img = mpimg.imread(image)
    #img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    width = img.shape[1]
    height = img.shape[0]
    return img, width, height


loadIMG('../data/train_images/*.jpg')
exit()
#original_image, width, height = IMG('../data/train_images/0010038bbd.jpg')


def display(img, modif, title):

    while True:
        
        cv2.imshow('origin', img)
        cv2.imshow(title, modif)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

img = grayscale(original_image)
display(original_image, img, '2gray')
#print(img.shape)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img



def equalize(img):
    img = cv2.equalizeHist(img)
    return img
img = equalize(img)
display(original_image, img, 'equalize')
#plt.imshow(img)
#print(img.shape)



#X_train = np.array(list(map(preprocessing, X_train)))
#X_test = np.array(list(map(preprocessing, X_test)))


def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image



def pan(image):
    pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image


def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image












## DATA AUGMENTATION

def random_augment(image):

    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)

    return image


def generate(images, count):
    generated = []
#     print(images)
    while True:
        for image in images:
            if len(generated) == count:
                return generated
            image = random_augment(image)
            generated.append(np.expand_dims(image, axis=2))

print(max(num_of_samples))


unique, counts = np.unique(y_train, return_counts=True)

target = max(num_of_samples)
x_augmented = []
y_augmented = []

for cls, count in tqdm(list(zip(unique, counts)), 'Augmenting training dataset'):
    diff = target - count
#     print(diff, target, count)
    x_augmented += generate(X_train[np.where(y_train == cls)[0]], diff)
    y_augmented += [cls for _ in range(diff)]



print(X_train.shape)
print(np.array(x_augmented).shape)
X_train = X_train.reshape(34799, 32, 32, 1)
X_valid = X_valid.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
x_train = np.concatenate([X_train, np.array(x_augmented)])
y_train = np.concatenate([y_train, np.array(y_augmented)])
n_train = y_train.size
print('Final number of training samples', n_train)


