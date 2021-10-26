import cv2 as cv
from PIL import Image
import numpy as np
import os
import sys

ROOT_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(ROOT_DIR, "images")
TIF_DIR = os.path.join(IMG_DIR, "tif")
JPG_DIR = os.path.join(IMG_DIR, "jpg")
OUT_DIR = os.path.join(ROOT_DIR, "output")

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xedn = 0
        self.ystart = 0
        self.yend = 0

    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]


def tif_to_jpg(tifpath):
    head_tail = list(os.path.split(tifpath))
    print(head_tail)
    if head_tail[1][-3:] == "tif":
        head_tail[1] = head_tail[1][:-4] + ".jpg"
    elif head_tail[1][-4:] == "tiff":
        head_tail[1] = head_tail[1][:-5] + ".jpg"
    else:
        print(f"Tif file {head_tail[1]} has unrecognized . extension (expected .tif or .tiff)")
        return 0
    outpath = os.path.join(JPG_DIR, head_tail[1])
    im = Image.open(tifpath)
    im.save(outpath, "JPEG", quality = 100)
    return outpath


def get_files(rpath):
    jpgs = []
    for root, dirs, files in os.walk(rpath):
        for f in files:
            if ".tif" in f:
                jpg_path = tif_to_jpg(os.path.join(root, f))
                jpgs.append(jpg_path)
    print(jpgs)
    return jpgs


def get_edges(infile, outfolder):
    head_tail = os.path.split(infile)
    outpath = os.path.join(OUT_DIR, head_tail[1][:-3] + "_edge.png")
    if os.path.isfile(outpath):
        return 0

    cap = cv.VideoCapture(infile)
    hasFrame, frame = cap.read()

    # Setting up layer
    try:
        cv.dnn_registerLayer('Crop', CropLayer)
    except:
        pass
    net = cv.dnn.readNet(os.path.abspath('deploy.prototxt'), os.path.abspath('hed_pretrained_bsds.caffemodel'))
    inp = cv.dnn.blobFromImage(frame, scalefactor = 1.0, crop = False)

    # Outputting from net...
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = 255 * out
    out = out.astype(np.uint8)
    cv.imshow("Display_Window", out)
    cv.imwrite(outpath, out)


def main():
    files = get_files(TIF_DIR)
    for f in files:
        get_edges(f, OUT_DIR)


if __name__ == "__main__":
    main()
