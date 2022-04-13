from __future__ import annotations
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import numpy as np
import cv2 as cv


class Mask(ABC):
    @abstractmethod
    def get_mask(self, size):
        ...

    @staticmethod
    def empty(size):
        return np.zeros((size[1], size[0]), np.uint8)


@dataclass_json
@dataclass
class Vec2:
    x: float
    y: float

    @staticmethod
    def from_tuple(tup):
        return Vec2(float(tup[0]), float(tup[1]))

    def as_tuple(self):
        return self.x, self.y

    def as_tensor(self):
        return torch.tensor([self.x, self.y])

    def rounded(self):
        return int(self.x), int(self.y)

    def linear_interpolation(
        self, other: Vec2, num: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.linspace(self.x, other.x, num), np.linspace(self.y, other.y, num)

    def translate(self, translation: Vec2) -> Vec2:
        return Vec2(self.x+translation.x, self.y+translation.y)
    
    def scale(self, scaling: float) -> Vec2:
        return Vec2(self.x*scaling, self.y*scaling)

    def __neg__(self):
        return Vec2(-self.x, -self.y)
    
    def dist(self, other: Vec2) -> float:
        return np.sqrt((self.x-other.x)**2+(self.y-other.y)**2)
    
    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __truediv__(self, div: int) -> Vec2:
        return Vec2(self.x/div, self.y/div)


@dataclass_json
@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_tuple(tup):
        return Vec2(float(tup[0]), float(tup[1]), float(tup[2]))

    def as_tuple(self):
        return self.x, self.y, self.z

    def rounded(self):
        return int(self.x), int(self.y), int(self.z)
    
    def vec2(self):
        return Vec2(self.x, self.y)

    def linear_interpolation(
        self, other: Vec3, num: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.linspace(self.x, other.x, num),
            np.linspace(self.y, other.y, num),
            np.linspace(self.z, other.z, num),
        )
    
    def translate(self, translation: Vec2) -> Vec3:
        return Vec3(self.x+translation.x, self.y+translation.y, self.z)
    
    def scale(self, scaling: float) -> Vec3:
        return Vec3(self.x*scaling, self.y*scaling, self.z*scaling)


@dataclass_json
@dataclass
class Quadratic(Mask):
    a: float
    b: float
    c: float

    def __call__(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    @staticmethod
    def from_points_precise(p1, p2, p3) -> Quadratic:
        x_in = np.array([p1[0], p2[0], p3[0]])
        mat = np.vstack((x_in ** 2, x_in, np.ones(3))).T
        y = np.array([p1[1], p2[1], p3[1]])

        solution = np.linalg.solve(mat, y)
        return Quadratic(solution[0], solution[1], solution[2])
    
    def visualise(self, img, color, line_thickness=3):
        x = np.arange(0, img.shape[1], 10)
        y = np.int32(np.clip(self(x), 0, img.shape[0]))
        pts = np.stack((x, y)).T
        img = cv.polylines(img, [pts], False, color, line_thickness)
        return img

    @staticmethod
    def from_points_least_sq(points) -> Quadratic:
        points = np.array(points)
        x_in = points[:, 0]
        mat = np.vstack((x_in ** 2, x_in, np.ones(len(points)))).T
        y = points[:, 1]

        solution = np.linalg.lstsq(mat, y, rcond=None)[0]
        return Quadratic(solution[0], solution[1], solution[2])

    def get_mask(self, size):
        mask = self.empty(size)
        for x in range(0, size[0]):
            mask[int(max(self(x), 0)) :, x] = 1
        return mask


@dataclass_json
@dataclass
class Ellipse(Mask):
    center: Vec2
    axes: Vec2
    angle: float

    @staticmethod
    def from_dict(data: dict):
        return Ellipse(
            Vec2(data["cx"], data["cy"]), Vec2(data["ax"], data["ay"]), data["angle"]
        )

    @staticmethod
    def from_points(points) -> Ellipse:
        if len(points) < 5:
            raise ValueError("Cannot infer ellipse from less than 5 points.")
        points = np.array(points, np.float32)
        center, axes, angle = cv.fitEllipse(points)
        return Ellipse(Vec2.from_tuple(center), Vec2.from_tuple(axes)/2, angle)

    def get_mask(self, size):
        return cv.ellipse(
            self.empty(size),
            self.center.rounded(),
            self.axes.rounded(),
            int(self.angle),
            0,
            360,
            1,
            -1,
        )

    def radius_at_angle(self, theta):
        a, b = self.axes.x, self.axes.y
        return (a * b) / np.sqrt(
            a ** 2 * np.sin(theta - self.angle) ** 2
            + b ** 2 * np.cos(theta - self.angle) ** 2
        )

    def intersect_angle(self, theta):
        r = self.radius_at_angle(theta)
        return Vec2(
            self.center.x + r * np.sin(theta), self.center.y + r * np.cos(theta)
        )

    def as_tuple(self):
        return self.center.as_tuple(), self.axes.as_tuple(), self.angle
    
    def visualise(self, img, color, line_thickness=3):
        img = cv.ellipse(img, self.center.rounded(), self.axes.rounded(), self.angle, 0, 360, color, line_thickness)
        return img
    
    def translate(self, translation: Vec2):
        return Ellipse(self.center.translate(translation), self.axes, self.angle)
    
    def scale(self, scaling: float):
        return Ellipse(self.center.scale(scaling), self.axes.scale(scaling), self.angle)
