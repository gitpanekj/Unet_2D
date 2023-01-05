from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile


class AGCallback(Callback):
    """ On given epoch end extracts feature maps from the end of
        attention gate structure and save those to a given directory.
    """

    def __init__(self, folder_name, data):
        super(AGCallback, self).__init__()
        self.folder_name = folder_name
        self.data = data
        print(os.getcwd())
        if os.getcwd().split("/")[-1] != folder_name:
            os.chdir(self.folder_name)

    def on_epoch_end(self, epoch, logs=None, period=10):
        if (epoch+1)%period == 0:
            os.mkdir(f"epoch_{epoch}")
            if len(self.data.shape) == 4:
                n = 0
                for layer in self.model.layers:
                    if "AG_upsampling" in layer.name:
                        partial_model = Model(inputs=self.model.inputs, outputs=layer.output)
                        wmap = partial_model.predict(self.data[np.newaxis, ...])
                        wmap = (wmap*255.).astype('uint8')
                        wmap = np.squeeze(wmap)
                        tifffile.imwrite(file=f"epoch_{epoch}/wmap-{n}.tif", data=wmap)
                        n += 1
                        wmap = 0
            else:
                    n = 0
                    for layer in self.model.layers:
                        if "AG_upsampling" in layer.name:
                            partial_model = Model(inputs=self.model.inputs, outputs=layer.output)
                            wmap = partial_model.predict(self.data[np.newaxis, ...])
                            self.plot_wmap(np.squeeze(wmap))
                            plt.savefig(f"epoch_{epoch}/wmap-{n}.png")
                            n += 1
                            plt.show()

    def plot_wmap(self, wmap):
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        im = plt.imshow(wmap, cmap='Reds')
        im.set_clim(0.,1.)
        plt.colorbar(cax=plt.axes([0.8, 0.1, 0.075, 0.8]))