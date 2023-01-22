from models.unet import Unet
from tensorflow.keras import Model, Input
import numpy as np
from tqdm import tqdm
from tensorflow.image import rgb_to_grayscale, resize
from tifffile import imread
import matplotlib.pyplot as plt
import os

GRAPH_PATH = "/home/panekj/Downloads/model.json"
WEIGHTS_PATH = "/home/panekj/Downloads/variables/variables"
SAMPLE_PATH = "/home/panekj/Downloads/155.tiff"

unet = Unet()
unet.load_model(GRAPH_PATH, WEIGHTS_PATH)

model = unet.model
del unet
    
    
data = imread(SAMPLE_PATH)
data = rgb_to_grayscale(resize(data, (800,1000)))
data = data[np.newaxis, ...]
    
# extract and save
index = 0
for layer in tqdm(model.layers):
    if 'relu' in layer.name:
            partial_model = Model(inputs=[model.inputs], outputs=[layer.output])
            pred = partial_model.predict(data)
            pred = np.squeeze(pred)
            del partial_model
            os.mkdir(f'/home/panekj/programming/Unet_2D/preds_3/{index}')
            for i in range(min(10, pred.shape[-1])):
                plt.imshow(pred[:,:,i], cmap='gray')
                plt.savefig(f"/home/panekj/programming/Unet_2D/preds_3/{index}/"+f'/{index}_{i}.png')
            del pred
            index += 1