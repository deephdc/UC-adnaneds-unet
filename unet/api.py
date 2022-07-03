# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing tasks.
In this way you don't mix your true code with DEEPaaS code and everything is more modular.
That is, if you need to write the predict() function in api.py, you would import your true predict
function and call it from here (with some processing/postprocessing in between if needed).
For example:

    import utils

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = utils.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""

from functools import wraps
import shutil
import tempfile

from aiohttp.web import HTTPBadRequest
from webargs import fields, validate


def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def get_metadata():
    metadata = {
        "author": "Adnane"
    }
    return metadata


def get_predict_args():
    """
    Input fields for the user.
    """
    arg_dict = {
        "demo-image": fields.Field(
            required=False,
            type="file",
            location="form",
            description="image",  # needed to be parsed by UI
        ),
        # Add format type of the response of predict()
        # For demo purposes, we allow the user to receive back
        # either an image or a zip containing an image.
        # More options for MIME types: https://mimeapplication.net/
        "accept": fields.Str(
            description="Media type(s) that is/are acceptable for the response.",
            validate=validate.OneOf(["image/*", "application/zip"]),
        ),
    }
    return arg_dict


import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from skimage import transform
from skimage import img_as_bool



IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

input_size_2 = (IMG_WIDTH, IMG_HEIGHT)
input_size_3 = (IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)



@_catch_error
def predict(**kwargs):
    """
    Return same inputs as provided.
    """
    filepath = kwargs['demo-image'].filename

    # Return the image directly
    if kwargs['accept'] == 'image/*':
        
        convert_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        convert_gray = np.asarray(convert_gray, dtype="uint8")
        mask_resized = cv2.resize(convert_gray, input_size_2, interpolation=cv2.INTER_NEAREST)
        
        return mask_resized
        #return open(filepath, 'rb')






    # Return a zip
    elif kwargs['accept'] == 'application/zip':

        zip_dir = tempfile.TemporaryDirectory()

        # Add original image to output zip
        shutil.copyfile(filepath,
                        zip_dir.name + '/demo.png')

        # Add for example a demo txt file
        with open(f'{zip_dir.name}/demo.txt', 'w') as f:
            f.write('Add here any additional information!')

        # Pack dir into zip and return it
        shutil.make_archive(zip_dir.name, format='zip', root_dir=zip_dir.name)
        zip_path = zip_dir.name + '.zip'

        return open(zip_path, 'rb')

# def get_metadata():
#     return {}
#
#
# def warm():
#     pass
#
#
# def get_predict_args():
#     return {}
#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None


################################################################
# Some functions that are not mandatory but that can be useful #
# (you can remove this if you don't need them)                 #
################################################################

# import pkg_resources
# import os


# BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))


# def _fields_to_dict(fields_in):
#     """
#     Function to convert mashmallow fields to dict()
#     """
#     dict_out = {}
#
#     for key, val in fields_in.items():
#         param = {}
#         param['default'] = val.missing
#         param['type'] = type(val.missing)
#         if key == 'files' or key == 'urls':
#             param['type'] = str
#
#         val_help = val.metadata['description']
#         if 'enum' in val.metadata.keys():
#             val_help = "{}. Choices: {}".format(val_help,
#                                                 val.metadata['enum'])
#         param['help'] = val_help
#
#         try:
#             val_req = val.required
#         except:
#             val_req = False
#         param['required'] = val_req
#
#         dict_out[key] = param
#     return dict_out
#
#
# def get_metadata():
#     """
#     Predefined get_metadata() that renders your module package configuration.
#     """
#
#     module = __name__.split('.', 1)
#
#     try:
#         pkg = pkg_resources.get_distribution(module[0])
#     except pkg_resources.RequirementParseError:
#         # if called from CLI, try to get pkg from the path
#         distros = list(pkg_resources.find_distributions(BASE_DIR,
#                                                         only=True))
#         if len(distros) == 1:
#             pkg = distros[0]
#     except Exception as e:
#         raise HTTPBadRequest(reason=e)
#
#     ### One can include arguments for train() in the metadata
#     train_args = _fields_to_dict(get_train_args())
#     # make 'type' JSON serializable
#     for key, val in train_args.items():
#         train_args[key]['type'] = str(val['type'])
#
#     ### One can include arguments for predict() in the metadata
#     predict_args = _fields_to_dict(get_predict_args())
#     # make 'type' JSON serializable
#     for key, val in predict_args.items():
#         predict_args[key]['type'] = str(val['type'])
#
#     meta = {
#         'name': None,
#         'version': None,
#         'summary': None,
#         'home-page': None,
#         'author': None,
#         'author-email': None,
#         'license': None,
#         'help-train': train_args,
#         'help-predict': predict_args
#     }
#
#     for line in pkg.get_metadata_lines("PKG-INFO"):
#         line_low = line.lower()  # to avoid inconsistency due to letter cases
#         for par in meta:
#             if line_low.startswith(par.lower() + ":"):
#                 _, value = line.split(": ", 1)
#                 meta[par] = value
#
#     return meta
