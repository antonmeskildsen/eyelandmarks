import os
from typing import List
from dataclasses_json import dataclass_json
import torch
from dataclasses import dataclass
from glob2 import glob

from torch.utils.data import Dataset

import cv2 as cv
import numpy as np
import pandas as pd

from PIL import Image
from torchvision.transforms import functional as T

from torchvision.transforms.transforms import RandomResizedCrop
from eyelandmarks.util.geometry import Vec2, Vec3, Ellipse


@dataclass_json
@dataclass
class EyeBall:
    pos: Vec2
    radius: float

    def map(self, f):
        return EyeBall(f(self.pos), self.radius)

    def translate(self, translation: Vec2):
        return self.map(lambda x: x.translate(translation))

    def scale(self, scaling: float):
        return self.map(lambda x: x.scale(scaling))


@dataclass_json
@dataclass
class Landmarks:
    points: List[Vec3]
    mean_accuracy: float

    def points_2d(self):
        return [p.vec2() for p in self.points]

    def points_2d_tuple(self):
        return [p.vec2().rounded() for p in self.points]

    def map(self, f):
        return Landmarks(list(map(f, self.points)), self.mean_accuracy)

    def visualise(self, img, color, marker, size, thickness):
        for v in self.points_2d():
            r = v.rounded()
            img = cv.drawMarker(img, r, color, marker, size, thickness)
        return img

    def translate(self, translation: Vec2):
        return self.map(lambda x: x.translate(translation))

    def scale(self, scaling: float):
        return self.map(lambda x: x.scale(scaling))

    def as_tensor(self):
        return torch.cat([x.vec2().as_tensor() for x in self.points])


@dataclass_json
@dataclass
class Pupil:
    valid: bool
    ellipse: Ellipse
    landmarks: Landmarks

    def translate(self, translation: Vec2):
        return Pupil(
            self.valid,
            self.ellipse.translate(translation),
            self.landmarks.translate(translation),
        )

    def scale(self, scaling: float):
        return Pupil(
            self.valid, self.ellipse.scale(scaling), self.landmarks.scale(scaling)
        )


@dataclass_json
@dataclass
class Iris:
    valid: bool
    ellipse: Ellipse
    landmarks: Landmarks

    def translate(self, translation: Vec2):
        return Iris(
            self.valid,
            self.ellipse.translate(translation),
            self.landmarks.translate(translation),
        )

    def scale(self, scaling: float):
        return Iris(
            self.valid, self.ellipse.scale(scaling), self.landmarks.scale(scaling)
        )


@dataclass_json
@dataclass
class EyeLid:
    valid: bool
    landmarks: Landmarks

    def translate(self, translation: Vec2):
        return EyeLid(self.valid, self.landmarks.translate(translation))

    def scale(self, scaling: float):
        return EyeLid(self.valid, self.landmarks.scale(scaling))


@dataclass_json
@dataclass
class Index:
    data_set: str
    video_id: int
    subject_id: int
    frame: int

    def file_identifier(self):
        return f"{self.data_set}_{self.subject_id}_{self.video_id}_{self.frame}"


@dataclass_json
@dataclass
class EyeSample:
    index: Index
    pupil: Pupil
    iris: Iris
    eyelid: EyeLid
    gaze: Vec2
    eyeball: EyeBall

    def visualise(self, img, thickness=3):
        img = img.copy()

        img = self.pupil.ellipse.visualise(img, (0, 0, 255), thickness)
        img = self.iris.ellipse.visualise(img, (0, 255, 0), thickness)
        img = self.pupil.landmarks.visualise(
            img, (255, 0, 0), cv.MARKER_DIAMOND, thickness // 2, thickness * 2
        )
        img = self.iris.landmarks.visualise(
            img, (255, 0, 0), cv.MARKER_DIAMOND, thickness // 2, thickness * 2
        )
        img = self.eyelid.landmarks.visualise(
            img, (255, 0, 0), cv.MARKER_DIAMOND, thickness // 2, thickness * 2
        )

        return img

    def translate(self, translation: Vec2):
        t = translation
        return EyeSample(
            self.index,
            self.pupil.translate(t),
            self.iris.translate(t),
            self.eyelid.translate(t),
            self.gaze.translate(t),
            self.eyeball,
        )

    def scale(self, scaling: float):
        s = scaling
        return EyeSample(
            self.index,
            self.pupil.scale(s),
            self.iris.scale(s),
            self.eyelid.scale(s),
            self.gaze.scale(s),
            self.eyeball,
        )


class EyeDataset(Dataset):
    def __init__(self, path: str, transformed_width=32, crop_size=250, random_crop=False,
                 random_scale=True):
        super().__init__()
        if not os.path.isdir(path):
            raise ValueError("Specified path cannot be found or is not a directory.")
        self.files = glob(os.path.join(path, "*.json"))
        self.path = path
        self.size = (288, 384)
        self.crop_size = crop_size
        self.transformed_width = transformed_width
        self.scale = transformed_width / crop_size
        self.random_crop = random_crop
        self.random_scale = random_scale

        self.crop_translation = Vec2(
            (self.size[1] - crop_size) / 2, (self.size[0] - crop_size) / 2
        )

    def __len__(self):
        return len(self.files)

    def target_transform(self, eye):
        ...

    def __getitem__(self, i):
        file = open(self.files[i])
        eye: EyeSample = EyeSample.from_json(file.read())
        file.close()

        img = cv.imread(os.path.join(self.path, eye.index.file_identifier() + ".jpg"))
        p_img = Image.fromarray(img)
        p_img = T.to_grayscale(p_img)
        p_tensor = T.to_tensor(p_img)

        if self.random_crop:
            # crop = transforms.RandomCrop(self.crop_size)

            i, j, h, w = RandomResizedCrop.get_params(p_tensor, scale=[0.5, 1],
                                                      ratio=[1, 1])

            crop_trans = Vec2(j, i)

            scale = 128 / h
            p_tensor = T.resized_crop(p_tensor, i, j, h, w, size=[128, 128])
        else:
            p_tensor = T.center_crop(p_tensor, [self.crop_size])
            crop_trans = self.crop_translation
            scale = self.scale

        p_tensor = T.resize(p_tensor, [self.transformed_width, self.transformed_width])

        eye = eye.translate(-crop_trans)
        eye = eye.scale(scale)

        return p_tensor, self.target_transform(eye)


class PupilCenterDataset(EyeDataset):
    def target_transform(self, eye):
        y = eye.pupil.ellipse.center.as_tensor() / self.transformed_width * 2 - 1
        return torch.tensor([y[0], y[1], int(eye.pupil.valid)])


class PupilLandmarksDataset(EyeDataset):
    def target_transform(self, eye):
        y = eye.pupil.landmarks.as_tensor() / self.transformed_width * 2 - 1
        return y


class AllLandmarksDataset(EyeDataset):
    def target_transform(self, eye):
        y = (
                torch.cat(
                    (
                        eye.pupil.landmarks.as_tensor(),
                        eye.iris.landmarks.as_tensor(),
                        eye.eyelid.landmarks.as_tensor(),
                    )
                )
                / self.transformed_width
                * 1  # Base image range is then -0.5, 0.5
                - 0.5
        )
        return y


class EyeRegDataset(Dataset):
    def __init__(self, path, transformed_width=32, crop_size=250, split="train"):
        super().__init__()
        self.iris = pd.read_csv(
            os.path.join(path, f"iris_{split}.csv")
        )  # Really pupil centre
        self.path = path
        self.size = (288, 384)
        self.crop_size = crop_size
        self.transformed_width = transformed_width
        self.scale = crop_size / transformed_width
        self.cropped_delta = (self.size[0] - crop_size) / 2, (
                self.size[1] - crop_size
        ) / 2

    def __len__(self):
        return len(self.iris)

    def __getitem__(self, i):
        row = self.iris.iloc[i]
        v_set = row["set"]
        subject_id = row["subject_id"]
        video_id = row["video_id"]
        f_index = row["frame"]

        img = cv.imread(
            os.path.join(self.path, f"{v_set}_{subject_id}_{video_id}_{f_index}.jpg")
        )

        p_img = Image.fromarray(img)
        p_img = T.to_grayscale(p_img)
        p_img = T.to_tensor(p_img)

        angle = (np.random.random() - 0.5) / 2
        angle_deg = angle / (np.pi * 2) * 360

        p_img = T.center_crop(p_img, [self.crop_size])
        p_img = T.resize(p_img, [self.transformed_width, self.transformed_width])
        p_img = T.rotate(p_img, -angle_deg)

        cx = (row["cx"] - self.cropped_delta[1]) / self.scale
        cy = (row["cy"] - self.cropped_delta[0]) / self.scale

        center = self.transformed_width / 2

        # R = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]

        cxx = (cx - center) * np.cos(angle) - (cy - center) * np.sin(angle)
        cyy = (cx - center) * np.sin(angle) + (cy - center) * np.cos(angle)

        return (
            p_img,
            torch.tensor([cxx + center, cyy + center]) / self.transformed_width * 2 - 1,
        )


class EyeSegDataset(Dataset):
    def __init__(self, path, transform, split="train"):
        super().__init__()
        self.iris = pd.read_csv(os.path.join(path, f"iris_{split}.csv"))
        self.path = path
        self.size = (288, 384)
        self.transform = transform

    def _gen_segmentation(self, cx, cy, ax, ay, angle):
        label = np.zeros(self.size)
        cv.ellipse(label, (cx, cy), (ax, ay), angle, 0, 360, 1, -1)
        return label

    def __len__(self):
        return len(self.iris)

    def __getitem__(self, i):
        row = self.iris.iloc[i]
        v_set = row["set"]
        subject_id = row["subject_id"]
        video_id = row["video_id"]
        f_index = row["frame"]

        img = cv.imread(
            os.path.join(self.path, f"{v_set}_{subject_id}_{video_id}_{f_index}.jpg")
        )
        label = self._gen_segmentation(
            int(row["cx"]),
            int(row["cy"]),
            int(row["width"] / 2),
            int(row["height"] / 2),
            row["angle"],
        )

        p_img = Image.fromarray(img)
        p_label = Image.fromarray(label)

        return self.transform(p_img), self.transform(p_label)
