from models.unet import Unet
from matplotlib.image import imread


u = Unet()
u.load_model("callbacks/network_graph.json", "/home/panekj/programming/Unet_2D/callbacks/weights/cp.ckpt")

paths = []
for path in paths:
    x = imread(path)
    y = u.model.predict(x)