import cv2
import os
import argparse
import numpy as np

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
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


def get_edges(infile, outpath, prototxt, model):
    if os.path.isfile(outpath):
        print(f"WARNING: CNN file output '{outpath}' already exists. Rewriting...")

    print(f"Performing CNN Canny on '{infile}'...")
    cap = cv2.VideoCapture(infile)
    hasFrame, frame = cap.read()

    # Setting up layer
    cv2.dnn_registerLayer('Crop', CropLayer)
    net = cv2.dnn.readNet(prototxt, model)
    inp = cv2.dnn.blobFromImage(frame, scalefactor = 1.0, crop = False)

    # Outputting from net...
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = 255 * out
    out = out.astype(np.uint8)
    cv2.imwrite(outpath, out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = 'The input file that should be processed')
    parser.add_argument('-o', '--output', help = 'The output file where the CNN Canny output should be saved')
    parser.add_argument('-p', '--prototxt', help = 'The prototxt file that should be used for computation')
    parser.add_argument('-m', '--model', help = 'The model file that should be used for CNN processing')
    args = parser.parse_args()
    get_edges(args.input, args.output, args.prototxt, args.model)

if __name__ == '__main__':
    main()
