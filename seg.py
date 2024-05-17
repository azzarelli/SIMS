# From https://www.kdnuggets.com/2021/10/real-time-image-segmentation-5-lines-code.html
# Checkpoint from https://github.com/ayoolaolafenwa/PixelLib/releases/download/0.2.0/pointrend_resnet50.pkl
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import numpy as np

ins = instanceSegmentation()
ins.load_model("./segmodels/pointrend_resnet50.pkl")
ins.segmentImage("test2.jpg",show_bboxes=True, output_image_name="output.jpg",
extract_segmented_objects= True, save_extracted_objects=True)


segmask, output = ins.segmentImage("test2.jpg")


# For the gegao scene the silly man in the back makes Gegao-detection challenging
gg = np.expand_dims(segmask['masks'][..., 0], axis=2)

cv2.imwrite("img.jpg", output)

mask = gg.squeeze().astype(np.uint8) * 255
cv2.imwrite("mask.jpg", mask)