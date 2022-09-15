#!/usr/bin/env python3

import os
import sys
import cv2
import datetime
import matplotlib as mp
mp.use('TkAgg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-image_path", type=str, required=True)
    parser.add_argument("-output_image_path", type=str, default="result.png")
    parser.add_argument("-num_test", type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())
    image = cv2.imread(args.image_path)

    start = datetime.datetime.now()
    for i in range(args.num_test):
        ou, overlay = segmentation_handler.run(image, only_mask=False)
    _processing_time = datetime.datetime.now() - start

    print(type(ou))

    plt.imshow(ou, interpolation='none')
    plt.show()

    print(BiSeNetV2Config())
    cv2.imshow("result", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(args.output_image_path, overlay)

    print("processing time one frame {}[ms]".format(_processing_time.total_seconds() * 1000 / args.num_test))


if __name__ == "__main__":
    main()
