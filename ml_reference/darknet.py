#!/usr/bin/env python
"""
Examples using darknet
"""
import os
import subprocess

DARKNET_PATH = '/Users/Tyler/Software_Development/darknet'

def yolo(image_filepath, path_to_darknet=DARKNET_PATH):
    args = ('./darknet', 'detect', 'cfg/yolov3.cfg', 'weights/yolov3.weights', image_filepath)

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=path_to_darknet)
    popen.wait()
    output = popen.stdout.read()

    # move the generated image to the local directory
    image_basename = os.path.splitext(os.path.basename(image_filepath))[0]
    args = ('mv', os.path.join(path_to_darknet, 'predictions.jpg'),
            '{}_predictions.jpg'.format(image_basename))
    subprocess.Popen(args)

    return output

def nightmare(image_filepath, layer=10, rounds=1, iters=10, range=1, octaves=4, rate=0.5, thresh=1.0,
              zoom=1.0, rotate=0.0,
              path_to_darknet=DARKNET_PATH):
    """
    rounds n: change the number of rounds. More rounds means more images generated
              and usually more change to the original image.
    iters n: change the number of iterations per round. More iterations means more
             change to the image per round.
    range n: change the range of possible layers. If set to one, only the given layer is
             chosen at every iteration. Otherwise, a layer is chosen randomly within than
             range (e.g. 10 -range 3 will choose between layers 9-11).
    octaves n: change the number of possible scales. At one octave, only the full size image
               is examined. Each additional octave adds a smaller version of the image
               (3/4 the size of the previous octave).
    rate x: change the learning rate for the image (default .05). Higher means more change to the
            image per iteration but also some instability and imprecision.
    thresh x: change the threshold for features to be magnified. Only features over x standard
              deviations away from the mean are magnified in the target layer. A higher threshold
              means fewer features are magnified.
    zoom x: change the zoom applied to the image after each round. You can optionally add a
            zoom in (x < 1) or zoom out (x > 1) to be applied to the image after each round.
    rotate x: change the rotation applied after each round. Optional rotation after each round.
    """
    args = ('./darknet', 'nightmare', 'cfg/vgg-conv.cfg', 'weights/vgg-conv.weights',
            image_filepath, '%d' % layer,
            '-rounds %d' % rounds, '-iters %d' % iters, '-range %d' % range, '-octaves %d' % octaves,
            '-rate %f' % rate, '-thresh %f' % thresh, '-zoom %f' % zoom, '-rotate %f' % rotate)

    popen = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=path_to_darknet)
    popen.wait()
    output = popen.stdout.read()

    # # TODO move the generated image to the local directory
    # image_basename = os.path.splitext(os.path.basename(image_filepath))[0]
    # args = ('mv', os.path.join(path_to_darknet, 'predictions.jpg'),
    #         '{}_predictions.jpg'.format(image_basename))
    # subprocess.Popen(args)

    return output

# image_filepath = os.path.join(path_to_darknet, 'data/scream.jpg')
image_filepath = '/Users/Tyler/Desktop/tonsai_coffee.jpg'

output = yolo(image_filepath)

# output = nightmare(image_filepath, 13)

print(output)
