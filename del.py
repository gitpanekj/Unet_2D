from models.unet import Unet

u = Unet()
u.load_model("callbacks/network_graph.json", "/home/panekj/programming/Unet_2D/callbacks/weights/cp.ckpt")