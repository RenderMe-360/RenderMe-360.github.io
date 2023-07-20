from calendar import c
from functools import partial
import json
from unittest.mock import NonCallableMagicMock

import time
import cv2
import h5py
import numpy as np
import torch
import tqdm
import sys


class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.actor_id = self.smc.attrs['actor_id']
        self.performance_part = self.smc.attrs['performance_part']
        self.capture_date = self.smc.attrs['capture_date']
        self.actor_info = dict(
            age=self.smc.attrs['age'],
            color=self.smc.attrs['color'],  # TODO
            gender=self.smc.attrs['gender'],
            height=self.smc.attrs['height'],  # TODO
            weight=self.smc.attrs['weight']  # TODO
        )
        self.Camera_info = dict(
            num_device=self.smc['Camera'].attrs['num_device'],
            num_frame=self.smc['Camera'].attrs['num_frame'],
            resolution=self.smc['Camera'].attrs['resolution'],
        )

    ###info
    def get_actor_info(self):
        return self.actor_info

    def get_Camera_info(self):
        return self.Camera_info

    ### Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self

        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict(
                Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'00' ... '59'}
                Matrix_type in ['D', 'K', 'RT']
        """
        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__
        self.__calibration_dict__ = dict()
        for ci in self.smc['Calibration'].keys():
            self.__calibration_dict__.setdefault(ci, dict())
            for mt in ['D', 'K', 'RT']:
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Calibration'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id

        Args:
            Camera_id (int/str of a number):
                CameraID(str) in {'00' ... '60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT']
        """
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc['Calibration'].keys(), f'Invalid Camera_id {Camera_id}'
        rs = dict()
        for k in ['D', 'K', 'RT']:
            rs[k] = self.smc['Calibration'][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_img(self, Camera_id, Image_type, Frame_id=None, disable_tqdm=True):
        """Get image its Camera_id, Image_type and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in
                    {'00'...'59'}
            Image_type(str) in
                    {'Camera': ['color','mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'-1
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
        """
        Camera_id = str(Camera_id)
        assert Camera_id in self.smc["Camera"].keys(), f'Invalid Camera_id {Camera_id}'
        assert Image_type in self.smc["Camera"][Camera_id].keys(), f'Invalid Image_type {Image_type}'
        assert isinstance(Frame_id, (list, int, str, type(None))), f'Invalid Frame_id datatype {type(Frame_id)}'
        if isinstance(Frame_id, (str, int)):
            Frame_id = str(Frame_id)
            assert Frame_id in self.smc["Camera"][Camera_id][Image_type].keys(), f'Invalid Frame_id {Frame_id}'
            if Image_type in ['color', 'mask']:
                img_byte = self.smc["Camera"][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_color = np.max(img_color, 2).astype(np.uint8)
            return img_color
        else:
            if Frame_id is None:
                Frame_id_list = sorted([int(l) for l in self.smc["Camera"][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_id, Image_type, fi))
            return np.stack(rs, axis=0)

    def get_audio(self):
        """
        Get audio data.
        Returns:
            a dictionary of audio data consists of:
                audio_np_array: np.ndarray
                sample_rate: int
        """
        if "s" not in self.performance_part.split('_')[0]:
            print(f"no audio data in the performance part: {self.performance_part}")
            return None
        data = self.smc["Camera"]['00']['audio']
        return data

    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id, Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in {18...32}
                    Not all the view have detection result, so the key will miss too when there are no lmk2d result
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            lmk2d
        """
        Camera_id = str(Camera_id)
        assert Camera_id in [f'%02d' % i for i in range(18, 33)], f'Invalid Camera_id {Camera_id}'
        assert isinstance(Frame_id, (list, int, str, type(None))), f'Invalid Frame_id datatype: {type(Frame_id)}'
        if Camera_id not in self.smc['Keypoints2d'].keys():
            print(f"not lmk2d result in camera id {Camera_id}")
            return None
        if isinstance(Frame_id, (str, int)):
            Frame_id = str(Frame_id)
            # assert Frame_id >= 0 and Frame_id<self.smc['Keypoints2d'].attrs['num_frame'], f'Invalid frame_index {Frame_id}'
            return self.smc['Keypoints2d'][Camera_id][Frame_id]
        else:
            if Frame_id is None:
                return self.smc['Keypoints2d'][Camera_id]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints2d(Camera_id, fi))
            return np.stack(rs, axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id, TODO coordinate

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
        """
        if isinstance(Frame_id, (str, int)):
            Frame_id = int(Frame_id)
            assert Frame_id >= 0 and Frame_id < self.smc['Keypoints3d'].attrs['num_frame'], \
                f'Invalid frame_index {Frame_id}'
            return self.smc['Keypoints3d'][str(Frame_id)]
        else:
            if Frame_id is None:
                return self.smc['Keypoints3d']
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints3d(fi))
            return np.stack(rs, axis=0)

