import numpy
import tensorflow as tf
import math
import argparse

from traffic_cnn import flatten_and_dense, default_model, medium_model, large_model

parser = argparse.ArgumentParser(description='Print model architectures')

parser.add_argument('model_number', metavar='M', type=int, nargs=1, help='model id either print model 0, 1 or 2')
parser.add_argument('--dense', dest='densify_bool', action='store_const', const=True, default=False, help='add --dense to add flattening and dense layers')

if __name__ == '__main__':
    print("\nHello Roni\n")
    args = parser.parse_args()

    txt = ["Default model for 32x32 images", "medium_model for 128x128 images", "Large_model for 200x200 images"]

    model_id = args.model_number[0]
    print(model_id)
    if model_id == 0:
        m = default_model()
    elif model_id == 1:
        m = medium_model()
    elif model_id == 2:
        m = large_model()

    if args.densify_bool:
        flatten_and_dense(m)

    m.summary()
    print(txt[model_id])
