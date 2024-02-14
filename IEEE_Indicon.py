
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'chest-xray-masks-and-labels:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F108201%2F258315%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240214%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240214T120212Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D5e314073a060e5a5a4e618b1930ce3b9da016e846bbbb314073f21a224b4e44712a5c9a8a078b88fdac8383e88618f0c3c1bf67a68c01c01c780860a0fdee1fa1b726d8434c80b150918b542516acea7fa7a35d80a0990bff6f3ce53530b94bd13759a3cfd0819a479843992f6edfb5708d7067b14db6b07a851ebc6211e333cdddca8b8513bc3d901549bf9dbbe87363a177fd6e070e1d06e56d973fdb15b30f4e09168a6a6739fdfc394ee386fc7532bfa114718a7c8fd2b94b47fd25480c339452a9400f4ae6fbd13532cc8ca0a3d3b18a0dd655aaae16af754e6ce39d3a5b831281907f9dc78a3fad0f658f123a041abd73d3682460d9f99f51fb0c8e81c'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

import numpy as np
import pandas as pd

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from glob import glob
from tensorflow import keras

from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.activations import *

lung_img_dir = '../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/'
mask_img_dir = '../input/chest-xray-masks-and-labels/Lung Segmentation/masks/'


lung_img_dir_glob = '../input/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/*.png'
mask_img_dir_glob = '../input/chest-xray-masks-and-labels/Lung Segmentation/masks/*.png'

lung_img = glob(lung_img_dir_glob)
mask_img = glob(mask_img_dir_glob)

len(lung_img), len(mask_img)
# We have 800 lung images but only 704 masks

# Create 1-1 relation
lungs_filenames = [name.split('/')[-1].split('.png')[0] for name in lung_img]
import re
import cv2
lung_paths = []
mask_paths = []
for lungs_filename in lungs_filenames:
    for mask_filename in mask_img:
        mask_match = re.search(lungs_filename, mask_filename)
        if mask_match:
            lung_paths.append(os.path.join(lung_img_dir,f"{lungs_filename}.png"))
            mask_paths.append(mask_filename)

assert len(mask_paths) == len(lung_paths)
print(f'Total number of samples {len(lung_paths)}')

"""### Load images"""

X = np.zeros((len(lung_paths),256,256))
y = np.zeros((len(lung_paths),256,256))

# Convert images to gray, resize and normalize for NN
def transform_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    reshape_img = cv2.resize(img_gray,img_shape)
    return reshape_img/255.0

# Loading images and masks
img_shape = (256,256)
for i, lung_path in tqdm(enumerate(lung_paths)):
    lung_img = cv2.imread(lung_path)
    X[i] = transform_img(lung_img)
for i , mask_path in tqdm(enumerate(mask_paths)):
    mask_img = cv2.imread(mask_path)
    y[i] = transform_img(mask_img)

def plot_mask(X,y):
    i,j = np.random.randint(0,len(X),2)
    plt.figure(figsize=(20,20))

    plt.subplot(121)
    plt.axis('off')
    plt.title('Chest X-ray with Mask')
    plt.imshow(np.hstack((X[i],y[i])), cmap = 'gray')
    plt.subplot(122)
    plt.axis('off')
    plt.title('Chest X-ray with Mask')
    plt.imshow(np.hstack((X[j],y[j])), cmap = 'gray')
    plt.savefig('sample.jpeg')
    plt.show()

plot_mask(X,y)

"""### U-net model"""

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu',dilation_rate=1, padding='same')(inputs)
    conv1= Conv2D(32,(3,3),activation="relu",dilation_rate=1,padding="same")(conv1)
    conv11 = Conv2D(32, (3, 3), activation='relu',dilation_rate=64, padding='valid')(conv1)


    conv2=  Conv2D(64,(3, 3),activation="relu",dilation_rate=1,padding="same")(conv11)
    conv2 = Conv2D(64, (3, 3), activation='relu',dilation_rate=1, padding='same')(conv2)
    conv21 = Conv2D(64, (3, 3), activation='relu',dilation_rate=32, padding='valid')(conv2)


    conv3=Conv2D(128,(3,3),activation="relu",dilation_rate=1,padding="same")(conv21)
    conv3 = Conv2D(128, (3, 3), activation='relu',dilation_rate=1,padding='same')(conv3)
    conv31 = Conv2D(128, (3, 3), activation='relu',dilation_rate=16, padding='valid')(conv3)


    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv31)
    conv4=Conv2D(256,(3,3),activation="relu",padding="same")(conv4)
    conv41 = Conv2D(256, (3, 3), activation='relu',dilation_rate=8,padding="valid")(conv4)


    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv41)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
# Bilinear interpolation followed by convolution
    biconv= UpSampling2D(size=(2,2),interpolation="bilinear")(conv5)
    convbi= Conv2D(256,(3,3),activation="relu",padding="same")(biconv)
    up1= concatenate([convbi,conv4],axis=3)
    convup1=Conv2D(256,(3,3),activation="relu",padding="same")(up1)
    convup1=Conv2D(256,(3,3),activation="relu",padding="same")(convup1)

#bilinear interpolation used for the decoder network followed by regular convolution
    biconv1=UpSampling2D(size=(2,2),interpolation="bilinear")(convup1)
    convbi1=Conv2D(128,(3,3),activation="relu",padding="same")(biconv1)
    up2= concatenate([convbi1,conv3],axis=3)
    convup2=Conv2D(128,(3,3),activation="relu",padding="same")(up2)
    convup2=Conv2D(128,(3,3),activation="relu",padding="same")(convup2)
#### binlinear interpolation for the decoder network followed by regular convolution
    biconv2=UpSampling2D(size=(2,2),interpolation="bilinear")(convup2)
    convbi2=Conv2D(64,(3,3),activation="relu",padding="same")(biconv2)
    up3 = concatenate([convbi2,conv2],axis=3)
    convup3=Conv2D(64,(3,3),activation="relu",padding="same")(up3)
    convup3=Conv2D(64,(3,3),activation="relu",padding="same")(convup3)
#### bilinear interpolation for the decoder network followed by the regular convolution
    biconv3=UpSampling2D(size=(2,2),interpolation="bilinear")(convup3)
    convbi3=Conv2D(32,(3,3),activation="relu",padding="same")(biconv3)
    up4 = concatenate([convbi3,conv1],axis=3)
    convup4=Conv2D(32,(3,3),activation="relu",padding="same")(up4)
    convup4=Conv2D(32,(3,3),activation="relu",padding="same")(convup4)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(convup4)
    return Model(inputs=[inputs], outputs=[conv10])

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = keras.sum(keras.abs(y_true * y_pred))
    sum_ = keras.sum(keras.square(y_true)) + keras.sum(keras.square(y_pred))
    jac = (intersection) / (sum_ - intersection)
    return jac

import tensorflow as tf
model = unet(input_size=(256,256,1))
model.compile(optimizer=Adam(lr=5*1e-4), loss="binary_crossentropy",
                  metrics=[dice_coef,iou,"binary_accuracy"])
model.summary()

"""### Callbacks"""

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="best_model.h5"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduce = ReduceLROnPlateau(monitor='val_loss',
                                   patience=5,
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=3)
callbacks = [checkpoint,reduce, early]

"""### Train Model"""

EPOCHS = 25
batch_size = 32
validation_spit = 0.2
history = model.fit(x = X,
                    y = y,
                    validation_split = validation_spit,
                    epochs = EPOCHS,
                    batch_size = batch_size,
                    callbacks=callbacks
                   )

fig, ax = plt.subplots(1, 2, figsize = (15, 4),dpi=200)

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

training_accuracy = history.history['binary_accuracy']
validation_accuracy = history.history['val_binary_accuracy']

epoch_count = range(1, len(training_loss) + 1)

ax[0].plot(epoch_count, training_loss, 'r-')
ax[0].plot(epoch_count, validation_loss, 'b--')
ax[0].legend(['Training Loss', 'Validation Loss'])
ax[0].set_title('Training vs Validation Loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epochs')

ax[1].plot(epoch_count, 100* np.array(training_accuracy), 'b--')
ax[1].plot(epoch_count, 100 *np.array(validation_accuracy), 'r-')
ax[1].legend(['Training Accuracy', 'Validation Accuracy'])
ax[1].set_title('Training vs Validation Accuracy')
ax[1].set_ylabel('Accuracy %')
ax[1].set_xlabel('Epochs')
plt.savefig('model.jpeg')

"""### Load test data"""

test_dir = '../input/chest-xray-masks-and-labels/Lung Segmentation/test/*'
test_img = glob(test_dir)

def add_mask(img,mask):
    _ , mask = cv2.threshold(mask ,0.3,1,cv2.THRESH_BINARY)
    merged = cv2.addWeighted(img, 0.7, mask.astype(np.float64), 0.3, 0)
    return merged

def plot_test(test_img,
              kernel =np.ones((5, 5),
                              np.uint8),
              save = False):
    size = 4
    test_size = len(test_img)
    test_sample = np.array(test_img)[np.random.randint(0, test_size, size).astype(int)]
    fig, axs = plt.subplots(nrows=size, ncols=4, figsize=(10, 12))
    cols = ['Real Images', 'Proposed Mask', 'Eroded Proposed Mask', 'Merged']

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)


    axs = axs.ravel()

    for i in range(16):

        if i %4 == 0:

            img = cv2.imread(test_sample[i//4])
            img = transform_img(img)
            axs[i].imshow(img, cmap = 'gray')

        elif i %4 == 1:
            mask = model.predict(np.array([img]))
            axs[i].imshow(mask[0], cmap = 'gray')

        elif i %4 == 2:
             erode_img = cv2.erode(mask[0], kernel, iterations=2)
             axs[i].imshow(erode_img, cmap = 'gray')
        else:
             merged = add_mask(img, erode_img)
             axs[i].imshow(merged, cmap = 'gray')
        axs[i].axis('off')

    if save:
        plt.savefig('predicted_mask.jpeg')
    fig.tight_layout()
    plt.show()

kernel =  kernel =np.ones((5, 5), np.uint8)
plot_test(test_img, save =True)

!pip freeze

