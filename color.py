
import pixellib
from pixellib.tune_bg import alter_bg



change_bg = alter_bg()
change_bg.load_pascalvoc_model("https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
change_bg.color_bg("tshirt/outfiter 8.JPG", colors = (0,128,0), output_image_name="colored_bg.jpg")