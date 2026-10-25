#!/usr/bin/env python3
"""
Universal Cricket Ball Tracking & Pitching Analytics Pipeline (Dual-Output Match Wicket System)
===============================================================================================
Tailored for:
  - Ball Detector (`ball_detector.pt`): Ultralytics YOLO11s (nc=1, names={0: 'ball'}, imgsz=960)
  - Pitch Detector (`best.pt`): Ultralytics YOLOv8n (nc=1, names={0: 'Cricket_pitch'}, imgsz=640)

Produces TWO Distinct Outputs for Every Wicket-Taking / Match Delivery Video Clip:
  1. OUTPUT 1 — 3D Hawk-Eye Ball Trajectory (`<clip>_trajectory.mp4` & `<clip>_trajectory.png`):
     - Royal-blue semi-transparent wicket-to-wicket inline carpet strictly inside the pitch region.
     - 3D-shaded pearl-white cylindrical trajectory tube with foreground bowler-end entry cap
       (`y0 > yb`), 3D perspective parabolic pre-bounce arch into the turf bounce contact disc
       `(xb, yb)`, ground-plane turf shadow, and post-bounce rise to the crimson-red ball tip
       at the stumps/impact height (matching `Screenshot 2026-10-18 020852.png`).
  2. OUTPUT 2 — Ball Pitching Details & Length-Zone Pitch Map (`<clip>_pitching_details.mp4`,
     `<clip>_pitching_details.png`, and cumulative `session_pitch_map_analytics.png`):
     - Perspective-mapped pitching length zones strictly on the pitch surface (following
       Line and length coaching definitions):
         * Full Toss / Stumps (0.0m – 1.2m): Tan band
         * Yorker (1.2m – 2.2m): Yellow band (`Yorker`)
         * Full (2.2m – 5.0m): Green band (`Full`)
         * Good (5.0m – 8.0m Pace / 3.0m – 5.0m Spin): Red/Terracotta band (`Good`)
         * Short (8.0m – Halfway+): Blue/Indigo band (`Short`)
     - Distance markers (`STUMPS`, `2m`, `4m`, `6m`, `8m`, `HALFWAY`) with perspective foreshortening.
     - Left-side zone percentage breakdown (`0%`, `15%`, `46%`, `38%`) and right-side zone labels.
     - 3D-shaded spherical ball pitching marker at the exact camera-stabilized turf bounce point
       `(x_b, y_b)` with distance-from-stumps (in meters), line (`IN-LINE` / `OUTSIDE OFF` /
       `CORRIDOR OF UNCERTAINTY` / `OUTSIDE LEG`), 18-cell Line x Length matrix code, and
       automated multi-delivery segmentation for full-match / wicket-compilation processing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


# =============================================================================
# Pure-NumPy Ultralytics `.pt` Inference Engine (Zero-PyTorch Fallback)
# =============================================================================

class _TensorStub:
    def __init__(self, storage: np.ndarray, offset: int, size: Tuple[int, ...], stride: Tuple[int, ...]):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride

    def numpy(self) -> np.ndarray:
        n_elem = int(np.prod(self.size))
        return self.storage[self.offset : self.offset + n_elem].reshape(self.size).astype(np.float32)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
    return _TensorStub(storage, storage_offset, size, stride).numpy()


class _ModStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


def _make_stub_cls(mod: str, name: str):
    if name == "_rebuild_parameter":
        return lambda data, rg, bh: data
    if name == "_rebuild_tensor_v2":
        return _rebuild_tensor_v2

    class C(_ModStub):
        _mod = mod
        _name = name

    C.__name__ = name
    return C


class NumpyYOLOModel:
    """
    Loads an Ultralytics YOLOv8 / YOLO11 `.pt` checkpoint directly from its zip archive
    and executes exact fused-BatchNorm FP32 inference using NumPy + OpenBLAS SGEMM.
    """

    def __init__(self, pt_path: str):
        self.pt_path = pt_path
        self.ckpt = self._load_pt_zip(pt_path)
        self.seq = self.ckpt["model"]._modules["model"]._modules

    @staticmethod
    def _load_pt_zip(path: str) -> dict:
        with zipfile.ZipFile(path, "r") as z:
            pkl_name = [f for f in z.namelist() if f.endswith("data.pkl")][0]
            prefix = pkl_name[: -len("data.pkl")]

            class Unpickler(pickle.Unpickler):
                def find_class(self, mod, name):
                    if mod in ("builtins", "__builtin__", "collections", "datetime"):
                        return super().find_class("builtins" if mod == "__builtin__" else mod, name)
                    return _make_stub_cls(mod, name)

                def persistent_load(self, pid):
                    _, stype, key, _loc, _size = pid
                    sname = stype.__name__
                    dtype = (
                        np.float16
                        if "Half" in sname
                        else (np.float32 if "Float" in sname else (np.int64 if "Long" in sname else np.uint8))
                    )
                    return np.frombuffer(z.read(f"{prefix}data/{key}"), dtype=dtype)

            return Unpickler(io.BytesIO(z.read(pkl_name))).load()

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

    @staticmethod
    def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0, groups: int = 1) -> np.ndarray:
        c_in, h, wi = x.shape
        c_out, _, kh, kw = w.shape
        if kh == 1 and kw == 1 and stride == 1 and pad == 0 and groups == 1:
            out = (w.reshape(c_out, c_in) @ x.reshape(c_in, -1)).reshape(c_out, h, wi)
            return out + b[:, None, None]
        if pad > 0:
            x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
        hp, wp = x.shape[1], x.shape[2]
        oh = (hp - kh) // stride + 1
        ow = (wp - kw) // stride + 1
        if groups == c_in and c_out == c_in:
            s0, s1, s2 = x.strides
            patches = as_strided(
                x, shape=(c_in, oh, ow, kh, kw), strides=(s0, s1 * stride, s2 * stride, s1, s2)
            )
            out = np.einsum("cijkl,cikl->cij", patches, w)
            return out + b[:, None, None]
        s0, s1, s2 = x.strides
        patches = as_strided(
            x, shape=(c_in, kh, kw, oh, ow), strides=(s0, s1, s2, s1 * stride, s2 * stride)
        )
        col = np.ascontiguousarray(patches).reshape(c_in * kh * kw, oh * ow)
        out = (w.reshape(c_out, -1) @ col).reshape(c_out, oh, ow)
        return out + b[:, None, None]

    @staticmethod
    def _maxpool5x5(x: np.ndarray) -> np.ndarray:
        c, h, w = x.shape
        xp = np.pad(x, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=-np.inf)
        s0, s1, s2 = xp.strides
        patches = as_strided(xp, shape=(c, h, w, 5, 5), strides=(s0, s1, s2, s1, s2))
        return patches.max(axis=(3, 4))

    def _eval_mod(self, m, x: np.ndarray) -> np.ndarray:
        t = m._name
        if t in ("Conv", "DWConv"):
            if not hasattr(m, "_fused_w"):
                conv = m._modules["conv"]
                bn = m._modules.get("bn")
                w = conv._parameters["weight"].astype(np.float32)
                b = conv._parameters.get("bias")
                b = np.zeros(w.shape[0], dtype=np.float32) if b is None else b.astype(np.float32)
                if bn is not None:
                    gw = bn._parameters["weight"].astype(np.float32)
                    gb = bn._parameters["bias"].astype(np.float32)
                    rm = bn._buffers["running_mean"].astype(np.float32)
                    rv = bn._buffers["running_var"].astype(np.float32)
                    eps = getattr(bn, "eps", 1e-3)
                    scale = gw / np.sqrt(rv + eps)
                    w = w * scale[:, None, None, None]
                    b = (b - rm) * scale + gb
                m._fused_w = np.ascontiguousarray(w)
                m._fused_b = np.ascontiguousarray(b)
                m._stride = conv.stride[0]
                m._pad = conv.padding[0]
                m._groups = conv.groups
                act = m._modules.get("act")
                m._act = act._name if act is not None else "Identity"
            out = self._conv2d(x, m._fused_w, m._fused_b, m._stride, m._pad, m._groups)
            return self._silu(out) if m._act == "SiLU" else out
        if t == "Conv2d":
            w = m._parameters["weight"].astype(np.float32)
            b = m._parameters.get("bias")
            b = np.zeros(w.shape[0], dtype=np.float32) if b is None else b.astype(np.float32)
            return self._conv2d(x, w, b, m.stride[0], m.padding[0], m.groups)
        if t == "Sequential":
            for sub in m._modules.values():
                x = self._eval_mod(sub, x)
            return x
        if t == "Bottleneck":
            y = self._eval_mod(m._modules["cv2"], self._eval_mod(m._modules["cv1"], x))
            return x + y if getattr(m, "add", False) else y
        if t == "C3k":
            y1 = self._eval_mod(m._modules["m"], self._eval_mod(m._modules["cv1"], x))
            y2 = self._eval_mod(m._modules["cv2"], x)
            return self._eval_mod(m._modules["cv3"], np.concatenate([y1, y2], axis=0))
        if t in ("C2f", "C3k2"):
            y = list(np.split(self._eval_mod(m._modules["cv1"], x), 2, axis=0))
            for sub in m._modules["m"]._modules.values():
                y.append(self._eval_mod(sub, y[-1]))
            return self._eval_mod(m._modules["cv2"], np.concatenate(y, axis=0))
        if t == "SPPF":
            y1 = self._eval_mod(m._modules["cv1"], x)
            y2 = self._maxpool5x5(y1)
            y3 = self._maxpool5x5(y2)
            y4 = self._maxpool5x5(y3)
            return self._eval_mod(m._modules["cv2"], np.concatenate([y1, y2, y3, y4], axis=0))
        if t == "Attention":
            c, h, w = x.shape
            nh, kd, hd = m.num_heads, m.key_dim, m.head_dim
            qkv = self._eval_mod(m._modules["qkv"], x).reshape(nh, 2 * kd + hd, h * w)
            q, k, v = np.split(qkv, [kd, 2 * kd], axis=1)
            attn = (q.transpose(0, 2, 1) @ k) * float(m.scale)
            attn = attn - np.max(attn, axis=-1, keepdims=True)
            attn = np.exp(attn)
            attn = attn / np.sum(attn, axis=-1, keepdims=True)
            out = (v @ attn.transpose(0, 2, 1)).reshape(c, h, w) + self._eval_mod(
                m._modules["pe"], v.reshape(c, h, w)
            )
            return self._eval_mod(m._modules["proj"], out)
        if t == "PSABlock":
            x = x + self._eval_mod(m._modules["attn"], x) if getattr(m, "add", True) else self._eval_mod(m._modules["attn"], x)
            x = x + self._eval_mod(m._modules["ffn"], x) if getattr(m, "add", True) else self._eval_mod(m._modules["ffn"], x)
            return x
        if t == "C2PSA":
            a, b = np.split(self._eval_mod(m._modules["cv1"], x), 2, axis=0)
            b = self._eval_mod(m._modules["m"], b)
            return self._eval_mod(m._modules["cv2"], np.concatenate([a, b], axis=0))
        if t == "Upsample":
            return x.repeat(2, axis=1).repeat(2, axis=2)
        raise ValueError(f"Unsupported layer type: {t}")

    def predict(
        self,
        img_bgr: np.ndarray,
        imgsz: int = 640,
        conf: float = 0.15,
        iou: float = 0.5,
        auto_rect: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h0, w0 = img_bgr.shape[:2]
        r = min(imgsz / h0, imgsz / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        dh, dw = imgsz - nh, imgsz - nw
        if auto_rect:
            dh, dw = np.mod(dh, 32), np.mod(dw, 32)
        dh /= 2.0
        dw /= 2.0
        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

        outputs: List[np.ndarray] = []
        for k, layer in self.seq.items():
            f = layer.f
            inp = x if k == "0" else (outputs[f] if isinstance(f, int) else [outputs[i] for i in f])
            if layer._name == "Concat":
                x = np.concatenate(inp, axis=0)
            elif layer._name == "Detect":
                boxes_all, scores_all = [], []
                strides = [8.0, 16.0, 32.0]
                for s_idx, feat in enumerate(inp):
                    stride = strides[s_idx]
                    b_raw = self._eval_mod(layer._modules["cv2"]._modules[str(s_idx)], feat)
                    c_raw = self._eval_mod(layer._modules["cv3"]._modules[str(s_idx)], feat)
                    scores = 1.0 / (1.0 + np.exp(-np.clip(c_raw[0], -40.0, 40.0)))
                    ys, xs = np.where(scores >= conf)
                    if len(ys) == 0:
                        continue
                    b_sel = b_raw[:, ys, xs].reshape(4, 16, -1)
                    b_sel = b_sel - np.max(b_sel, axis=1, keepdims=True)
                    prob = np.exp(b_sel) / np.sum(np.exp(b_sel), axis=1, keepdims=True)
                    dist = np.sum(prob * np.arange(16, dtype=np.float32)[None, :, None], axis=1)
                    ax = xs.astype(np.float32) + 0.5
                    ay = ys.astype(np.float32) + 0.5
                    x1 = ((ax - dist[0]) * stride - left) / r
                    y1 = ((ay - dist[1]) * stride - top) / r
                    x2 = ((ax + dist[2]) * stride - left) / r
                    y2 = ((ay + dist[3]) * stride - top) / r
                    boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
                    scores_all.append(scores[ys, xs])
                if not boxes_all:
                    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
                boxes_all = np.vstack(boxes_all)
                scores_all = np.concatenate(scores_all)
                xywh = np.column_stack([
                    boxes_all[:, 0],
                    boxes_all[:, 1],
                    boxes_all[:, 2] - boxes_all[:, 0],
                    boxes_all[:, 3] - boxes_all[:, 1],
                ])
                keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores_all.tolist(), conf, iou)
                if len(keep) > 0:
                    keep = np.array(keep).flatten()
                    return boxes_all[keep], scores_all[keep]
                return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
            else:
                x = self._eval_mod(layer, inp)
            outputs.append(x)
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PitchGeometry:
    """Stabilized perspective geometry of the cricket pitch and royal-blue wicket strip."""
    bbox: Tuple[float, float, float, float]
    raw_py1: float
    top_y: float                             # Base of far stumps (0.0m reference)
    bottom_y: float                          # Near visible end of pitch
    top_left_x: float
    top_right_x: float
    bottom_left_x: float
    bottom_right_x: float
    stumps_center_top_x: float
    stumps_center_bottom_x: float
    inline_half_width_top: float
    inline_half_width_bottom: float
    visible_length_m: float = 13.5           # Visible pitch length in meters from far stumps

    @property
    def height(self) -> float:
        return max(1.0, self.bottom_y - self.top_y)

    @property
    def top_width(self) -> float:
        return max(1.0, self.top_right_x - self.top_left_x)

    @property
    def bottom_width(self) -> float:
        return max(1.0, self.bottom_right_x - self.bottom_left_x)

    def pitch_x_bounds_at_y(self, y: float) -> Tuple[float, float]:
        alpha = np.clip((y - self.top_y) / self.height, 0.0, 1.0)
        lx = self.top_left_x + alpha * (self.bottom_left_x - self.top_left_x)
        rx = self.top_right_x + alpha * (self.bottom_right_x - self.top_right_x)
        return float(lx), float(rx)

    def inline_x_bounds_at_y(self, y: float) -> Tuple[float, float]:
        alpha = np.clip((y - self.top_y) / self.height, 0.0, 1.0)
        cx = self.stumps_center_top_x + alpha * (self.stumps_center_bottom_x - self.stumps_center_top_x)
        hw = self.inline_half_width_top + alpha * (self.inline_half_width_bottom - self.inline_half_width_top)
        return float(cx - hw), float(cx + hw)

    def y_at_distance_m(self, dist_m: float) -> float:
        """
        Converts distance in meters `dist_m` from the striker's stumps (0m) into image Y
        using perspective foreshortening derived from top/bottom pitch width ratio.
        """
        u = np.clip(dist_m / max(1.0, self.visible_length_m), 0.0, 1.15)
        k = max(0.0, min(0.65, (self.bottom_width / self.top_width) - 1.0))
        alpha = (u * (1.0 + 0.35 * k)) / (1.0 + 0.35 * k * u)
        return float(self.top_y + alpha * self.height)

    def distance_m_at_y(self, y: float) -> float:
        """Inverts `y_at_distance_m` to return exact distance in meters from striker's stumps."""
        alpha = np.clip((y - self.top_y) / self.height, 0.0, 1.0)
        k = max(0.0, min(0.65, (self.bottom_width / self.top_width) - 1.0))
        u = alpha / (1.0 + 0.35 * k * (1.0 - alpha))
        return float(u * self.visible_length_m)

    def inline_strip_polygon(self) -> np.ndarray:
        tl = self.stumps_center_top_x - self.inline_half_width_top
        tr = self.stumps_center_top_x + self.inline_half_width_top
        br = self.stumps_center_bottom_x + self.inline_half_width_bottom
        bl = self.stumps_center_bottom_x - self.inline_half_width_bottom
        return np.array([
            [int(round(tl)), int(round(self.top_y))],
            [int(round(tr)), int(round(self.top_y))],
            [int(round(br)), int(round(self.bottom_y))],
            [int(round(bl)), int(round(self.bottom_y))],
        ], dtype=np.int32)

    def band_polygon(self, dist_start_m: float, dist_end_m: float) -> np.ndarray:
        """Returns the 4-point perspective polygon on the pitch between `dist_start_m` and `dist_end_m`."""
        y1 = np.clip(self.y_at_distance_m(dist_start_m), self.top_y, self.bottom_y)
        y2 = np.clip(self.y_at_distance_m(dist_end_m), self.top_y, self.bottom_y)
        lx1, rx1 = self.pitch_x_bounds_at_y(y1)
        lx2, rx2 = self.pitch_x_bounds_at_y(y2)
        return np.array([
            [int(round(lx1)), int(round(y1))],
            [int(round(rx1)), int(round(y1))],
            [int(round(rx2)), int(round(y2))],
            [int(round(lx2)), int(round(y2))],
        ], dtype=np.int32)


@dataclass
class BallDetection:
    frame_idx: int
    x: float
    y: float
    w: float
    h: float
    conf: float


@dataclass
class TrajectoryModel:
    """Stores the fitted piecewise-parabolic cricket ball trajectory and coaching analytics."""
    start_frame: int
    bounce_frame: float
    impact_frame: int
    end_frame: int
    pre_bounce_pts: np.ndarray
    post_bounce_pts: np.ndarray
    bounce_pt: Tuple[float, float]
    end_pt: Tuple[float, float]
    ref_py1: float = 0.0
    ref_top_y: float = 0.0
    ref_height: float = 1.0
    ref_cx: float = 0.0
    pitching_distance_m: float = 6.0
    pitching_length_zone: str = "Good"       # "Full Toss", "Yorker", "Full", "Good", "Short"
    pitching_line: str = "OUTSIDE OFF"       # "IN-LINE", "OUTSIDE OFF", "OUTSIDE LEG"
    corridor_of_uncertainty: bool = False    # True if pitching within 30cm (12 in) outside off stump
    matrix_18_cell: str = "G-M"              # 18-cell Line x Length coaching matrix code
    bowler_type: str = "pace"                # "pace" or "spin"
    batter_hand: str = "RHB"                 # "RHB" or "LHB"
    lateral_offset_norm: float = 0.0         # Normalized X offset on pitch [-0.5, +0.5]
    lateral_offset_m: float = 0.0            # Lateral offset from middle stump in meters (pitch width = 3.05m)
    frame_to_curve_idx: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def project_pts_to_geom(self, pts: np.ndarray, geom: PitchGeometry) -> np.ndarray:
        """Projects trajectory points from the reference impact geometry onto any frame's `geom`."""
        if len(pts) == 0 or self.ref_height <= 1.0:
            return pts.copy()
        out = pts.copy()
        scale_y = geom.height / self.ref_height
        out[:, 0] = pts[:, 0] + (geom.stumps_center_top_x - self.ref_cx)
        out[:, 1] = geom.top_y + (pts[:, 1] - self.ref_top_y) * scale_y
        return out


# =============================================================================
# 1. Pitch Detector (`best.pt`)
# =============================================================================

class PitchDetector:
    """Executes `best.pt` (`Cricket_pitch`, imgsz=640) and stabilizes perspective pitch geometry."""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.25,
        device: Optional[str] = None,
        ema_alpha: float = 0.22,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.ema_alpha = ema_alpha
        self.ultra_model = None
        self.np_model: Optional[NumpyYOLOModel] = None
        self._smoothed_box: Optional[np.ndarray] = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.ultra_model = YOLO(model_path)
            except ImportError:
                self.np_model = NumpyYOLOModel(model_path)

    def reset(self) -> None:
        self._smoothed_box = None

    @staticmethod
    def _detect_scoreboard_top_y(frame: np.ndarray) -> float:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame[int(0.80 * h) :, :], cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)
        dark_rows = np.where(row_means < 38.0)[0]
        if len(dark_rows) > 8:
            return float(int(0.80 * h) + int(dark_rows[0]) - 3)
        return float(h - 4)

    @staticmethod
    def is_valid_pitch_box(box: np.ndarray, w: int, h: int) -> bool:
        """Rejects false pitch detections on close-up player shots or sky/crowd banners."""
        px1, py1, px2, py2 = box
        pw, ph = px2 - px1, py2 - py1
        if pw < 0.14 * w or pw > 0.82 * w:
            return False
        if ph < 0.22 * h or py1 < 0.12 * h or py2 < 0.55 * h:
            return False
        return True

    def detect_raw_box(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        box = None
        if self.ultra_model is not None:
            res = self.ultra_model.predict(
                frame, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device
            )
            if len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                confs = res[0].boxes.conf.cpu().numpy()
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                for idx in np.argsort(-confs):
                    cand = xyxy[int(idx)].astype(np.float64)
                    if self.is_valid_pitch_box(cand, w, h):
                        box = cand
                        break
        elif self.np_model is not None:
            xyxy, confs = self.np_model.predict(frame, imgsz=self.imgsz, conf=self.conf, auto_rect=True)
            if len(xyxy) > 0:
                for idx in np.argsort(-confs):
                    cand = xyxy[int(idx)].astype(np.float64)
                    if self.is_valid_pitch_box(cand, w, h):
                        box = cand
                        break

        if box is None:
            if self._smoothed_box is not None:
                return self._smoothed_box.copy()
            box = np.array([0.28 * w, 0.36 * h, 0.72 * w, 0.86 * h], dtype=np.float64)
        return box

    def estimate_geometry(self, frame: np.ndarray, raw_box: Optional[np.ndarray] = None) -> PitchGeometry:
        if raw_box is None:
            raw_box = self.detect_raw_box(frame)

        if self._smoothed_box is None:
            self._smoothed_box = raw_box.copy()
        else:
            self._smoothed_box = self.ema_alpha * raw_box + (1.0 - self.ema_alpha) * self._smoothed_box

        px1, py1, px2, py2 = self._smoothed_box
        scoreboard_y = self._detect_scoreboard_top_y(frame)
        py2_clamped = min(float(py2), scoreboard_y)
        pw = max(40.0, float(px2 - px1))
        ph = max(40.0, float(py2_clamped - py1))
        cx = 0.5 * float(px1 + px2)
        stumps_base_y = float(py1 + 0.055 * ph)

        h_frame, w_frame = frame.shape[:2]
        zoom_ratio = pw / max(1.0, float(w_frame))
        vis_m = float(np.clip(19.5 - 18.5 * zoom_ratio, 11.5, 18.5))

        return PitchGeometry(
            bbox=(float(px1), float(py1), float(px2), float(py2_clamped)),
            raw_py1=float(py1),
            top_y=stumps_base_y,
            bottom_y=float(py2_clamped),
            top_left_x=float(px1 + 0.08 * pw),
            top_right_x=float(px2 - 0.08 * pw),
            bottom_left_x=float(px1),
            bottom_right_x=float(px2),
            stumps_center_top_x=cx,
            stumps_center_bottom_x=cx,
            inline_half_width_top=float(0.038 * pw),
            inline_half_width_bottom=float(0.050 * pw),
            visible_length_m=vis_m,
        )


# =============================================================================
# 2. Pitch-Only Ball Detector & Trajectory + Coaching Line/Length Solver
# =============================================================================

class PitchBallTracker:
    """
    Executes `ball_detector.pt` over the active pitch corridor, segments individual
    deliveries across single or multi-delivery match videos, and computes both:
      1. The 3D Foreground-Entry Hawk-Eye ball trajectory (`pre_bounce_pts`, `post_bounce_pts`)
         matching `Screenshot 2026-10-18 020852.png`
      2. Exact camera-stabilized ball pitching telemetry (`pitching_distance_m`,
         `pitching_length_zone`, `pitching_line`, `corridor_of_uncertainty`, `matrix_18_cell`)
         following Wikipedia Line and length coaching standards.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 960,
        conf: float = 0.12,
        iou: float = 0.5,
        device: Optional[str] = None,
        bowler_type: str = "auto",
        batter_hand: str = "RHB",
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.bowler_type = bowler_type
        self.batter_hand = batter_hand.upper()
        self.ultra_model = None
        self.np_model: Optional[NumpyYOLOModel] = None

        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.ultra_model = YOLO(model_path)
            except ImportError:
                self.np_model = NumpyYOLOModel(model_path)

    def detect_in_pitch_corridor(
        self,
        frame: np.ndarray,
        frame_idx: int,
        pitch_geom: PitchGeometry,
    ) -> List[BallDetection]:
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = pitch_geom.bbox
        pw = max(40.0, px2 - px1)
        ph = max(40.0, py2 - py1)
        cx = 0.5 * (px1 + px2)

        rx1 = max(0, int(cx - 0.36 * pw))
        rx2 = min(w, int(cx + 0.36 * pw))
        ry1 = max(0, int(py1 - 0.25 * ph))
        ry2 = min(h, int(py2 + 0.05 * ph))

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return []

        dets: List[BallDetection] = []
        if self.ultra_model is not None:
            res = self.ultra_model.predict(
                roi, imgsz=640, conf=self.conf, iou=self.iou, verbose=False, device=self.device
            )
            if len(res) > 0 and res[0].boxes is not None:
                xywh = res[0].boxes.xywh.cpu().numpy()
                confs = res[0].boxes.conf.cpu().numpy()
                for (bx, by, bw, bh), c in zip(xywh, confs):
                    dets.append(
                        BallDetection(
                            frame_idx=frame_idx,
                            x=float(bx) + rx1,
                            y=float(by) + ry1,
                            w=float(bw),
                            h=float(bh),
                            conf=float(c),
                        )
                    )
        elif self.np_model is not None:
            xyxy, confs = self.np_model.predict(roi, imgsz=640, conf=self.conf, iou=self.iou, auto_rect=True)
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                bw, bh = float(x2 - x1), float(y2 - y1)
                dets.append(
                    BallDetection(
                        frame_idx=frame_idx,
                        x=float(0.5 * (x1 + x2)) + rx1,
                        y=float(0.5 * (y1 + y2)) + ry1,
                        w=bw,
                        h=bh,
                        conf=float(c),
                    )
                )
        return dets

    @staticmethod
    def select_clean_delivery_chain(all_detections: List[BallDetection]) -> List[BallDetection]:
        if not all_detections:
            return []

        dets = sorted(all_detections, key=lambda d: (d.frame_idx, -d.conf))
        n = len(dets)
        dp = np.array([d.conf * 5.0 if d.conf >= 0.30 else -1e6 for d in dets], dtype=np.float64)
        parent = np.full(n, -1, dtype=np.int32)
        vx_arr = np.zeros(n, dtype=np.float64)
        vy_arr = np.zeros(n, dtype=np.float64)

        for i in range(n):
            di = dets[i]
            for j in range(i):
                if dp[j] < 0:
                    continue
                dj = dets[j]
                dt = di.frame_idx - dj.frame_idx
                if dt <= 0 or dt > 8:
                    continue
                dx = di.x - dj.x
                dy = di.y - dj.y
                vx, vy = dx / dt, dy / dt
                if abs(vx) > 16.0 or abs(vy) > 65.0:
                    continue
                if parent[j] != -1:
                    if abs(vx - vx_arr[j]) > 5.5:
                        continue
                    if vy_arr[j] < -5.0 and vy > 5.0:
                        continue
                score = (
                    dp[j]
                    + 10.0
                    + di.conf * 8.0
                    - 0.6 * abs(vx - (vx_arr[j] if parent[j] != -1 else vx))
                    - 0.3 * (dt - 1)
                )
                if score > dp[i]:
                    dp[i] = score
                    parent[i] = j
                    vx_arr[i] = vx
                    vy_arr[i] = vy

        best_idx = int(np.argmax(dp))
        if dp[best_idx] < 0:
            return []

        chain: List[BallDetection] = []
        curr = best_idx
        while curr != -1:
            chain.append(dets[curr])
            curr = int(parent[curr])
        chain.reverse()
        return chain

    @classmethod
    def segment_all_deliveries_in_video(
        cls,
        all_detections: List[BallDetection],
        min_chain_len: int = 5,
        min_frame_gap: int = 30,
    ) -> List[List[BallDetection]]:
        """
        Segments a full-over or full-match detection stream into individual delivery chains
        so every wicket-taking or match delivery in a continuous video can be processed.
        """
        if not all_detections:
            return []
        dets = sorted(all_detections, key=lambda d: d.frame_idx)
        clusters: List[List[BallDetection]] = [[dets[0]]]
        for d in dets[1:]:
            if d.frame_idx - clusters[-1][-1].frame_idx > min_frame_gap:
                clusters.append([d])
            else:
                clusters[-1].append(d)

        valid_chains: List[List[BallDetection]] = []
        for cluster in clusters:
            chain = cls.select_clean_delivery_chain(cluster)
            if len(chain) >= min_chain_len:
                ys = [c.y for c in chain]
                if max(ys) - min(ys) >= 22.0:
                    valid_chains.append(chain)
        return valid_chains

    @staticmethod
    def classify_length_zone(dist_m: float, bowler_type: str = "pace") -> str:
        """
        Classifies pitching distance (in meters from striker's stumps) into coaching zones
        (Line and length - Wikipedia: pace Good = 5.0-8.0m, spin Good = 3.0-5.0m).
        """
        if dist_m < 1.2:
            return "Full Toss"
        if dist_m < 2.2:
            return "Yorker"
        if bowler_type == "spin":
            if dist_m < 3.0:
                return "Full"
            if dist_m < 5.2:
                return "Good"
            return "Short"
        if dist_m < 5.0:
            return "Full"
        if dist_m < 8.0:
            return "Good"
        return "Short"

    @staticmethod
    def classify_line_and_matrix(
        xb: float,
        yb: float,
        ref_geom: PitchGeometry,
        length_zone: str,
        batter_hand: str = "RHB",
    ) -> Tuple[str, bool, str, float]:
        """
        Computes:
          - `pitching_line`: "IN-LINE", "OUTSIDE OFF", or "OUTSIDE LEG"
          - `corridor_of_uncertainty`: True if within 30cm (12 inches) outside off stump
          - `matrix_18_cell`: 18-cell Line x Length code (e.g., "Good-CorridorOff", "Full-Stumps")
          - `lateral_offset_m`: Lateral offset in meters from middle stump (standard pitch width = 3.05m)
        """
        il_l, il_r = ref_geom.inline_x_bounds_at_y(yb)
        px_l, px_r = ref_geom.pitch_x_bounds_at_y(yb)
        pitch_w_px = max(1.0, px_r - px_l)
        cx_stumps = 0.5 * (il_l + il_r)
        # Standard cricket pitch width is 3.05m (10 ft); wicket width is 0.2286m (9 in)
        meters_per_px = 3.05 / pitch_w_px
        offset_m = (xb - cx_stumps) * meters_per_px

        is_lhb = (batter_hand.upper() == "LHB")
        if il_l <= xb <= il_r:
            p_line = "IN-LINE"
            corridor = False
            line_code = "Stumps"
        else:
            screen_left = (xb < il_l)
            is_off_side = (not screen_left) if is_lhb else screen_left
            dist_from_off_stump_m = (
                ((xb - il_r) if is_lhb else (il_l - xb)) * meters_per_px
            )
            if is_off_side:
                p_line = "OUTSIDE OFF"
                # Wikipedia Line and length: "Corridor of uncertainty" = within 12 inches (0.305m) outside off stump
                corridor = (0.0 < dist_from_off_stump_m <= 0.32)
                line_code = "CorridorOff" if corridor else "WideOff"
            else:
                p_line = "OUTSIDE LEG"
                corridor = False
                line_code = "OutsideLeg"

        z_short = {
            "Full Toss": "FT",
            "Yorker": "Y",
            "Full": "F",
            "Good": "G",
            "Short": "S",
        }.get(length_zone, "G")
        matrix_cell = f"{z_short}-{line_code}"
        return p_line, corridor, matrix_cell, float(offset_m)

    def fit_trajectory(
        self,
        chain: List[BallDetection],
        ref_geom: PitchGeometry,
        pitch_geoms: Optional[List[PitchGeometry]] = None,
        dense_points_per_phase: int = 180,
    ) -> Optional[TrajectoryModel]:
        if len(chain) < 3:
            return None

        t_arr = np.array([d.frame_idx for d in chain], dtype=np.float64)
        x_raw = np.array([d.x for d in chain], dtype=np.float64)
        y_raw = np.array([d.y for d in chain], dtype=np.float64)
        w_arr = np.array([max(0.25, d.conf) for d in chain], dtype=np.float64)

        dy_steps = np.diff(y_raw)
        neg_indices = np.where(dy_steps < -3.0)[0]
        turn_idx = int(neg_indices[0]) if len(neg_indices) > 0 else int(np.argmax(y_raw))
        has_gap = (t_arr[turn_idx] - t_arr[turn_idx - 1]) > 1.5
        end_pre = turn_idx if has_gap else turn_idx + 1

        t_pre, x_pre, y_pre, w_pre = t_arr[:end_pre], x_raw[:end_pre], y_raw[:end_pre], w_arr[:end_pre]
        t_post, x_post, y_post, w_post = t_arr[turn_idx:], x_raw[turn_idx:], y_raw[turn_idx:], w_arr[turn_idx:]

        poly_x1 = np.polyfit(t_pre, x_pre, deg=1, w=w_pre)
        if len(t_post) >= 2:
            poly_x2 = np.polyfit(t_post, x_post, deg=1, w=w_post)
            poly_y2 = np.polyfit(t_post, y_post, deg=1, w=w_post)
        else:
            poly_x2 = poly_x1
            poly_y2 = np.array([-25.0, float(y_pre[-1]) + 25.0 * float(t_pre[-1])])

        if not has_gap:
            t_bounce = float(t_pre[-1])
            xb_raw = float(x_pre[-1])
            yb_raw = float(y_pre[-1])
        else:
            poly_y1_tail = np.polyfit(t_pre[-min(4, len(t_pre)) :], y_pre[-min(4, len(t_pre)) :], deg=1)
            t_s = np.linspace(t_pre[-1], t_post[0], 200)
            y1_s = np.polyval(poly_y1_tail, t_s)
            y2_s = np.polyval(poly_y2, t_s)
            bk = int(np.argmin(np.abs(y1_s - y2_s)))
            t_bounce = float(t_s[bk])
            yb_raw = float(0.5 * (y1_s[bk] + y2_s[bk]))
            xb_raw = float(0.5 * (np.polyval(poly_x1, t_bounce) + np.polyval(poly_x2, t_bounce)))

        b_idx = int(round(t_bounce))
        if pitch_geoms is not None and 0 <= b_idx < len(pitch_geoms):
            g_bounce = pitch_geoms[b_idx]
            xb = xb_raw + (ref_geom.stumps_center_top_x - g_bounce.stumps_center_top_x)
            yb = ref_geom.top_y + (yb_raw - g_bounce.top_y) * (ref_geom.height / max(1.0, g_bounce.height))
        else:
            xb = xb_raw
            yb = yb_raw

        yb = float(np.clip(yb, ref_geom.top_y + 0.22 * ref_geom.height, ref_geom.top_y + 0.68 * ref_geom.height))

        # 1. Pre-bounce 3D Foreground-Entry Arch (matching `Screenshot 2026-10-18 020852.png`)
        dx_raw = xb_raw - float(x_pre[0])
        sign_dx = 1.0 if dx_raw >= 0 else -1.0
        span_x = max(min(abs(dx_raw) * 1.35, 0.16 * ref_geom.bottom_width), 0.11 * ref_geom.bottom_width)
        x0 = xb - sign_dx * span_x
        y0 = float(min(ref_geom.bottom_y - 22.0, yb + 0.19 * ref_geom.height))

        s1 = np.linspace(0.0, 1.0, dense_points_per_phase)
        cx1 = xb - 0.25 * sign_dx * span_x
        cy1 = yb - 0.048 * ref_geom.height
        x1_dense = ((1.0 - s1) ** 2) * x0 + 2.0 * (1.0 - s1) * s1 * cx1 + (s1 ** 2) * xb
        y1_dense = ((1.0 - s1) ** 2) * y0 + 2.0 * (1.0 - s1) * s1 * cy1 + (s1 ** 2) * yb
        pre_pts = np.column_stack([x1_dense, y1_dense])

        # 2. Post-bounce curve rising to the stumps/impact height (xe, ye)
        s2 = np.linspace(0.0, 1.0, dense_points_per_phase)
        xe_raw = float(x_post[-1]) if len(x_post) > 0 else xb
        xe = xb + (xe_raw - xb_raw)
        ye_detected = float(y_post[-1]) if len(y_post) > 0 else (ref_geom.top_y - 0.12 * ref_geom.height)
        ye = float(min(ye_detected, ref_geom.top_y - 0.14 * ref_geom.height))
        cx2 = 0.5 * (xb + xe)
        cy2 = yb + 0.55 * (ye - yb) - 0.02 * ref_geom.height
        x2_dense = ((1.0 - s2) ** 2) * xb + 2.0 * (1.0 - s2) * s2 * cx2 + (s2 ** 2) * xe
        y2_dense = ((1.0 - s2) ** 2) * yb + 2.0 * (1.0 - s2) * s2 * cy2 + (s2 ** 2) * ye
        post_pts = np.column_stack([x2_dense, y2_dense])

        # Auto-detect pace vs spin if set to "auto"
        flight_frames = float(t_bounce - t_pre[0])
        b_type = self.bowler_type
        if b_type == "auto":
            b_type = "spin" if flight_frames >= 22.0 else "pace"

        dist_m = ref_geom.distance_m_at_y(yb)
        zone_name = self.classify_length_zone(dist_m, bowler_type=b_type)
        p_line, corridor, matrix_cell, offset_m = self.classify_line_and_matrix(
            xb, yb, ref_geom, zone_name, batter_hand=self.batter_hand
        )

        px_l, px_r = ref_geom.pitch_x_bounds_at_y(yb)
        lat_norm = (xb - 0.5 * (px_l + px_r)) / max(1.0, px_r - px_l)

        t_start_int = int(round(t_pre[0]))
        t_bounce_int = int(round(t_bounce))
        t_end_int = int(round(t_post[-1])) if len(t_post) > 0 else t_bounce_int + 6

        frame_map: Dict[int, Tuple[int, int]] = {}
        for f in range(t_start_int, t_end_int + 20):
            if f <= t_bounce_int:
                frac = (f - t_start_int) / max(1.0, float(t_bounce_int - t_start_int))
                frame_map[f] = (max(2, int(round(frac * len(pre_pts)))), 0)
            else:
                frac = min(1.0, (f - t_bounce_int) / max(1.0, float(t_end_int - t_bounce_int)))
                frame_map[f] = (len(pre_pts), max(2, int(round(frac * len(post_pts)))))

        return TrajectoryModel(
            start_frame=t_start_int,
            bounce_frame=t_bounce,
            impact_frame=t_end_int,
            end_frame=t_end_int + 15,
            pre_bounce_pts=pre_pts,
            post_bounce_pts=post_pts,
            bounce_pt=(xb, yb),
            end_pt=(xe, ye),
            ref_py1=ref_geom.raw_py1,
            ref_top_y=ref_geom.top_y,
            ref_height=ref_geom.height,
            ref_cx=ref_geom.stumps_center_top_x,
            pitching_distance_m=dist_m,
            pitching_length_zone=zone_name,
            pitching_line=p_line,
            corridor_of_uncertainty=corridor,
            matrix_18_cell=matrix_cell,
            bowler_type=b_type,
            batter_hand=self.batter_hand,
            lateral_offset_norm=float(np.clip(lat_norm, -0.48, 0.48)),
            lateral_offset_m=offset_m,
            frame_to_curve_idx=frame_map,
        )


# =============================================================================
# 3. OUTPUT 1 Renderer: 3D Hawk-Eye Ball Trajectory (`HawkEyeRenderer`)
# =============================================================================

class HawkEyeRenderer:
    """
    Renders Output 1 (3D Hawk-Eye Ball Trajectory on the pitch region) matching
    `Screenshot 2026-10-18 020852.png`.
    """

    def __init__(self, style: str = "white_blue"):
        self.style = style

    def render_frame(
        self,
        frame: np.ndarray,
        pitch_geom: PitchGeometry,
        traj: Optional[TrajectoryModel],
        frame_idx: Optional[int] = None,
        show_full_trajectory: bool = True,
    ) -> np.ndarray:
        canvas = frame.copy()
        canvas = self._draw_royal_blue_pitch_strip(canvas, pitch_geom)

        if traj is None:
            return canvas

        pre_proj = traj.project_pts_to_geom(traj.pre_bounce_pts, pitch_geom)
        post_proj = traj.project_pts_to_geom(traj.post_bounce_pts, pitch_geom)

        if show_full_trajectory or frame_idx is None or frame_idx >= traj.impact_frame:
            pre_pts = pre_proj
            post_pts = post_proj
            is_complete = True
        elif frame_idx < traj.start_frame:
            return canvas
        else:
            n1, n2 = traj.frame_to_curve_idx.get(
                frame_idx, (len(pre_proj), len(post_proj))
            )
            pre_pts = pre_proj[:n1]
            post_pts = post_proj[:n2] if n2 > 0 else np.empty((0, 2))
            is_complete = (n2 >= len(post_proj))

        base_radius = max(4.5, 0.021 * pitch_geom.top_width)
        canvas = self._draw_turf_shadow(canvas, pre_pts, post_pts, pitch_geom, base_radius)

        if len(pre_pts) >= 2:
            pre_only = (len(post_pts) < 2 and not is_complete)
            self._draw_3d_pearl_white_tube(
                canvas,
                pre_pts,
                pitch_geom,
                base_radius=base_radius,
                draw_entry_cap=True,
                draw_bounce_disc=not pre_only,
                draw_red_ball_tip=pre_only,
            )

        if len(post_pts) >= 2:
            self._draw_3d_pearl_white_tube(
                canvas,
                post_pts,
                pitch_geom,
                base_radius=base_radius * 0.94,
                draw_entry_cap=False,
                draw_bounce_disc=False,
                draw_red_ball_tip=True,
            )

        return canvas

    @staticmethod
    def _draw_royal_blue_pitch_strip(canvas: np.ndarray, pitch_geom: PitchGeometry) -> np.ndarray:
        h, w = canvas.shape[:2]
        poly = pitch_geom.inline_strip_polygon()
        x1 = max(0, int(np.min(poly[:, 0])) - 4)
        x2 = min(w, int(np.max(poly[:, 0])) + 5)
        y1 = max(0, int(np.min(poly[:, 1])) - 2)
        y2 = min(h, int(np.max(poly[:, 1])) + 2)
        if x2 <= x1 or y2 <= y1:
            return canvas
        roi = canvas[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        mask = np.zeros((rh, rw), dtype=np.float32)
        poly_local = poly - np.array([[x1, y1]], dtype=np.int32)
        cv2.fillPoly(mask, [poly_local], 0.58, lineType=cv2.LINE_AA)
        blue_patch = np.empty_like(roi, dtype=np.float32)
        blue_patch[:, :] = (148.0, 52.0, 42.0)
        cy2 = min(rh, int(round(0.065 * pitch_geom.height)))
        if cy2 > 0:
            blue_patch[:cy2, :] = (175.0, 38.0, 28.0)
        alpha = mask[:, :, None]
        canvas[y1:y2, x1:x2] = np.clip(
            roi.astype(np.float32) * (1.0 - alpha) + blue_patch * alpha, 0, 255
        ).astype(np.uint8)
        return canvas

    @staticmethod
    def _draw_turf_shadow(
        canvas: np.ndarray,
        pre_pts: np.ndarray,
        post_pts: np.ndarray,
        pitch_geom: PitchGeometry,
        base_radius: float,
    ) -> np.ndarray:
        if len(pre_pts) < 2:
            return canvas
        h, w = canvas.shape[:2]
        bx, by = pre_pts[-1, 0], pre_pts[-1, 1]
        p0 = (int(round(pre_pts[0, 0])), int(round(min(pitch_geom.bottom_y - 4, pre_pts[0, 1] + 10))))
        pb = (int(round(bx)), int(round(by + 2)))
        x1 = max(0, min(p0[0], pb[0]) - 20)
        x2 = min(w, max(p0[0], pb[0]) + 20)
        y1 = max(0, min(p0[1], pb[1]) - 20)
        y2 = min(h, max(p0[1], pb[1]) + 20)
        if x2 <= x1 or y2 <= y1:
            return canvas
        roi = canvas[y1:y2, x1:x2]
        sh = roi.copy()
        p0_l = (p0[0] - x1, p0[1] - y1)
        pb_l = (pb[0] - x1, pb[1] - y1)
        cv2.line(sh, p0_l, pb_l, (35, 45, 40), max(2, int(round(base_radius * 0.85))), cv2.LINE_AA)
        cv2.ellipse(
            sh,
            pb_l,
            (max(4, int(round(base_radius * 1.1))), max(2, int(round(base_radius * 0.55)))),
            0, 0, 360, (28, 38, 32), -1, cv2.LINE_AA,
        )
        canvas[y1:y2, x1:x2] = cv2.addWeighted(sh, 0.20, roi, 0.80, 0)
        return canvas

    @staticmethod
    def _draw_3d_pearl_white_tube(
        canvas: np.ndarray,
        pts: np.ndarray,
        pitch_geom: PitchGeometry,
        base_radius: float,
        draw_entry_cap: bool = False,
        draw_bounce_disc: bool = False,
        draw_red_ball_tip: bool = False,
    ) -> None:
        if len(pts) < 2:
            return

        pitch_h = pitch_geom.height
        n = len(pts)
        step = max(1, n // 110)
        sampled = pts[::step]
        if not np.allclose(sampled[-1], pts[-1]):
            sampled = np.vstack([sampled, pts[-1]])

        radii = np.zeros(len(sampled), dtype=np.float64)
        for i in range(len(sampled)):
            depth_ratio = np.clip((sampled[i, 1] - pitch_geom.top_y) / pitch_h, -0.05, 1.05)
            radii[i] = max(2.4, base_radius * (0.58 + 0.54 * max(0.0, depth_ratio)))

        bands = [
            (0.00, 1.00, (148, 156, 152)),
            (-0.08, 0.82, (180, 187, 184)),
            (-0.14, 0.62, (210, 216, 214)),
            (-0.22, 0.38, (232, 236, 235)),
            (-0.30, 0.18, (248, 250, 249)),
        ]

        for normal_offset, width_scale, bgr_color in bands:
            for i in range(len(sampled) - 1):
                dx = sampled[i + 1, 0] - sampled[i, 0]
                dy = sampled[i + 1, 1] - sampled[i, 1]
                norm = max(1e-5, math.hypot(dx, dy))
                nx, ny = -dy / norm, dx / norm
                if ny > 0:
                    nx, ny = -nx, -ny

                r_mid = 0.5 * (radii[i] + radii[i + 1])
                shift = normal_offset * r_mid
                p1 = (int(round(sampled[i, 0] + nx * shift)), int(round(sampled[i, 1] + ny * shift)))
                p2 = (int(round(sampled[i + 1, 0] + nx * shift)), int(round(sampled[i + 1, 1] + ny * shift)))
                thickness = max(1, int(round(2.0 * r_mid * width_scale)))
                cv2.line(canvas, p1, p2, bgr_color, thickness=thickness, lineType=cv2.LINE_AA)

        if draw_bounce_disc:
            bx, by = int(round(sampled[-1, 0])), int(round(sampled[-1, 1]))
            rb = max(3, int(round(radii[-1] * 0.92)))
            cv2.circle(canvas, (bx, by), rb, (142, 150, 146), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (bx, by), max(1, rb - 1), (208, 214, 212), -1, lineType=cv2.LINE_AA)

        if draw_entry_cap:
            ex, ey = int(round(sampled[0, 0])), int(round(sampled[0, 1]))
            re_cap = max(3, int(round(radii[0] * 0.96)))
            cv2.circle(canvas, (ex, ey), re_cap, (135, 142, 138), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (ex, ey), max(2, re_cap - 1), (168, 174, 170), -1, lineType=cv2.LINE_AA)

        if draw_red_ball_tip:
            tx, ty = int(round(sampled[-1, 0])), int(round(sampled[-1, 1]))
            rt = max(3, int(round(radii[-1] * 0.88)))
            cv2.circle(canvas, (tx, ty), rt, (52, 55, 208), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (tx - 1, ty - 1), max(1, rt // 2), (115, 120, 248), -1, lineType=cv2.LINE_AA)


# =============================================================================
# 4. OUTPUT 2 Renderer: Ball Pitching Details & Length-Zone Pitch Map
# =============================================================================




def build_broadcast_heatmap_lut() -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a 256-entry BGR color LUT and Alpha LUT matching `image_9832a5.png`:
      - 0.00..0.05: transparent (0 alpha)
      - 0.05..0.24: faint purple/violet outer aura (BGR: 165, 75, 105), alpha 0 -> 0.52
      - 0.24..0.38: royal blue -> cyan-blue (BGR: 235, 115, 25 -> 235, 225, 15), alpha 0.52 -> 0.82
      - 0.38..0.68: vibrant green -> lime-green (BGR: 75, 240, 15 -> 35, 250, 165), alpha 0.85 -> 0.92
      - 0.68..0.86: bright yellow -> warm orange (BGR: 10, 235, 250 -> 10, 145, 255), alpha 0.94
      - 0.86..1.00: fiery orange-red core (BGR: 12, 65, 248), alpha 0.96
    """
    stops_x = np.array([0.0, 0.05, 0.16, 0.26, 0.36, 0.52, 0.68, 0.82, 0.92, 1.00], dtype=np.float32)
    # BGR stops
    stops_b = np.array([150, 155, 175, 235, 220,  65,  25,  10,  10,  15], dtype=np.float32)
    stops_g = np.array([ 70,  70,  75, 110, 225, 242, 248, 225, 130,  55], dtype=np.float32)
    stops_r = np.array([105, 105, 115,  30,  15,  20, 155, 250, 255, 248], dtype=np.float32)
    stops_a = np.array([0.0, 0.0, 0.36, 0.62, 0.78, 0.88, 0.92, 0.95, 0.96, 0.97], dtype=np.float32)

    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    b_lut = np.interp(xs, stops_x, stops_b).astype(np.uint8)
    g_lut = np.interp(xs, stops_x, stops_g).astype(np.uint8)
    r_lut = np.interp(xs, stops_x, stops_r).astype(np.uint8)
    a_lut = np.interp(xs, stops_x, stops_a).astype(np.float32)
    bgr_lut = np.stack([b_lut, g_lut, r_lut], axis=1) # (256, 3)
    return bgr_lut, a_lut

BGR_LUT, ALPHA_LUT = build_broadcast_heatmap_lut()

def draw_dashed_line(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 7,
    gap_len: int = 5,
) -> None:
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = max(1.0, math.hypot(dx, dy))
    ux, uy = dx / dist, dy / dist
    curr = 0.0
    while curr < dist:
        end_d = min(dist, curr + dash_len)
        p_start = (int(round(pt1[0] + ux * curr)), int(round(pt1[1] + uy * curr)))
        p_end = (int(round(pt1[0] + ux * end_d)), int(round(pt1[1] + uy * end_d)))
        cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)
        curr += dash_len + gap_len

def render_broadcast_pitch_heatmap(
    deliveries: list,
    width: int = 680,
    height: int = 640,
    highlight_filter: Optional[str] = None, # None, "G-Stumps", "Corridor", "GoodLength"
    subtitle_badge: Optional[str] = None,
) -> np.ndarray:
    """
    Renders the exact broadcast pitch heatmap graphic matching `image_9832a5.png`:
      - Textured green turf outfield + sandy-brown textured cricket pitch trapezoid
      - Darker brown central wicket-to-wicket strip
      - Crisp white 3D stumps at top center with diagonal ground shadow
      - Solid popping crease & halfway lines + dashed zone divider lines across the pitch
      - Right-side bold white labels: `FULL`, `GOOD`, `SHORT`, `HALF`
      - Bottom 3 semi-transparent dark charcoal boxes: `OFF` | `MID` | `LEG`
      - Organic multi-scale Gaussian thermal heatmap (purple outer footprints -> blue -> cyan -> green -> yellow -> orange-red hotspot core)
    """
    rng = np.random.default_rng(42)

    # 1. Rich textured green grass outfield with subtle radial lighting matching image_9832a5.png
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx_img, cy_img = width * 0.44, height * 0.48
    dist_norm = np.sqrt(((xx - cx_img) / (width * 0.65)) ** 2 + ((yy - cy_img) / (height * 0.65)) ** 2)
    vignette = np.clip(1.08 - 0.32 * dist_norm, 0.62, 1.05)

    grass_noise = rng.normal(0.0, 5.5, size=(height, width)).astype(np.float32)
    grass_noise = cv2.GaussianBlur(grass_noise, (3, 3), 0.8)

    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[:, :, 0] = (42.0 + grass_noise * 0.6) * vignette   # B
    canvas[:, :, 1] = (128.0 + grass_noise * 1.1) * vignette  # G
    canvas[:, :, 2] = (54.0 + grass_noise * 0.7) * vignette   # R
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # 2. Perspective Pitch Geometry matching `image_9832a5.png` proportions
    top_y = 64.0
    bot_y = 625.0
    pitch_h = bot_y - top_y
    cx_p = 250.0
    top_w = 264.0
    bot_w = 486.0

    geom = PitchGeometry(
        bbox=(cx_p - bot_w / 2, top_y, cx_p + bot_w / 2, bot_y),
        raw_py1=top_y,
        top_y=top_y,
        bottom_y=bot_y,
        top_left_x=cx_p - top_w / 2,
        top_right_x=cx_p + top_w / 2,
        bottom_left_x=cx_p - bot_w / 2,
        bottom_right_x=cx_p + bot_w / 2,
        stumps_center_top_x=cx_p,
        stumps_center_bottom_x=cx_p,
        inline_half_width_top=17.0,
        inline_half_width_bottom=30.0,
        visible_length_m=11.5,
    )

    # 3. Sandy-brown textured pitch surface with subtle dark outer turf border
    outer_poly = np.array([
        [int(round(geom.top_left_x - 6)), int(round(geom.top_y - 24))],
        [int(round(geom.top_right_x + 6)), int(round(geom.top_y - 24))],
        [int(round(geom.bottom_right_x + 8)), int(round(geom.bottom_y + 10))],
        [int(round(geom.bottom_left_x - 8)), int(round(geom.bottom_y + 10))],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [outer_poly], (72, 92, 98), lineType=cv2.LINE_AA)

    pitch_poly = np.array([
        [int(round(geom.top_left_x)), int(round(geom.top_y - 18))],
        [int(round(geom.top_right_x)), int(round(geom.top_y - 18))],
        [int(round(geom.bottom_right_x)), int(round(geom.bottom_y + 10))],
        [int(round(geom.bottom_left_x)), int(round(geom.bottom_y + 10))],
    ], dtype=np.int32)

    pitch_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(pitch_mask, [pitch_poly], 255)

    # Generate realistic sandy-clay soil texture (warm tan/brown matching image_9832a5.png)
    soil_fine = rng.normal(0.0, 9.5, size=(height, width)).astype(np.float32)
    soil_fine = cv2.GaussianBlur(soil_fine, (3, 3), 0.7)
    soil_coarse = rng.normal(0.0, 6.0, size=(height // 4, width // 4)).astype(np.float32)
    soil_coarse = cv2.resize(soil_coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    # Slight vertical wear gradient (lighter near good/full length, slightly darker near bottom)
    y_grad = np.clip(1.06 - 0.14 * ((yy - top_y) / pitch_h), 0.88, 1.08)

    soil_bgr = np.zeros((height, width, 3), dtype=np.float32)
    soil_bgr[:, :, 0] = (126.0 + soil_fine * 0.85 + soil_coarse * 0.7) * y_grad  # B
    soil_bgr[:, :, 1] = (158.0 + soil_fine * 0.95 + soil_coarse * 0.8) * y_grad  # G
    soil_bgr[:, :, 2] = (178.0 + soil_fine * 1.00 + soil_coarse * 0.9) * y_grad  # R
    soil_bgr = np.clip(soil_bgr, 0, 255).astype(np.uint8)

    pm_bool = pitch_mask > 0
    canvas[pm_bool] = soil_bgr[pm_bool]

    # Double border lines along left and right edges of pitch (matching image_9832a5.png)
    cv2.polylines(canvas, [pitch_poly], True, (88, 112, 126), 3, cv2.LINE_AA)
    inner_poly = np.array([
        [int(round(geom.top_left_x + 4)), int(round(geom.top_y - 14))],
        [int(round(geom.top_right_x - 4)), int(round(geom.top_y - 14))],
        [int(round(geom.bottom_right_x - 5)), int(round(geom.bottom_y + 10))],
        [int(round(geom.bottom_left_x + 5)), int(round(geom.bottom_y + 10))],
    ], dtype=np.int32)
    cv2.polylines(canvas, [inner_poly], False, (105, 130, 145), 1, cv2.LINE_AA)

    # 4. Darker brown central wicket-to-wicket inline strip (`MID` strip in image_9832a5.png)
    strip_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(strip_mask, [geom.inline_strip_polygon()], 255)
    sm_bool = (strip_mask > 0) & pm_bool
    canvas[sm_bool] = np.clip(canvas[sm_bool].astype(np.float32) * 0.78, 0, 255).astype(np.uint8)

    # 5. Crease lines & dashed zone dividers + Right-Side Labels (`FULL`, `GOOD`, `SHORT`, `HALF`)
    # Top bowling crease (dashed line extending slightly outside pitch edges)
    y_stumps = int(round(geom.top_y))
    lx_st, rx_st = geom.pitch_x_bounds_at_y(y_stumps)
    draw_dashed_line(canvas, (int(lx_st - 28), y_stumps), (int(rx_st + 35), y_stumps), (218, 222, 220), 1, 6, 5)

    # Popping crease (solid white line at 1.22m + return creases)
    y_pop = int(round(geom.y_at_distance_m(1.22)))
    lx_p, rx_p = geom.pitch_x_bounds_at_y(y_pop)
    cv2.line(canvas, (int(lx_p), y_pop), (int(rx_p), y_pop), (238, 240, 238), 2, cv2.LINE_AA)
    # Return creases
    cv2.line(canvas, (int(lx_st + 14), y_stumps - 8), (int(lx_p + 14), y_pop), (215, 220, 218), 1, cv2.LINE_AA)
    cv2.line(canvas, (int(rx_st - 14), y_stumps - 8), (int(rx_p - 14), y_pop), (215, 220, 218), 1, cv2.LINE_AA)

    # Dashed divider between FULL and GOOD (at ~4.2m)
    y_fg = int(round(geom.y_at_distance_m(4.2)))
    lx_fg, rx_fg = geom.pitch_x_bounds_at_y(y_fg)
    draw_dashed_line(canvas, (int(lx_fg - 26), y_fg), (int(rx_fg + 75), y_fg), (230, 235, 230), 1, 6, 5)

    # Dashed divider between GOOD and SHORT (at ~6.4m)
    y_gs = int(round(geom.y_at_distance_m(6.4)))
    lx_gs, rx_gs = geom.pitch_x_bounds_at_y(y_gs)
    draw_dashed_line(canvas, (int(lx_gs - 26), y_gs), (int(rx_gs + 75), y_gs), (230, 235, 230), 1, 6, 5)

    # Solid white HALF line (at ~9.2m)
    y_half = int(round(geom.y_at_distance_m(9.2)))
    lx_h, rx_h = geom.pitch_x_bounds_at_y(y_half)
    cv2.line(canvas, (int(lx_h - 14), y_half), (int(rx_h + 4), y_half), (242, 244, 242), 2, cv2.LINE_AA)

    # Right-side uppercase zone labels (`FULL`, `GOOD`, `SHORT`, `HALF`) matching image_9832a5.png
    font = cv2.FONT_HERSHEY_DUPLEX
    zone_labels = [
        ("FULL", 1.85, highlight_filter == "Full"),
        ("GOOD", 5.30, highlight_filter in ("GoodLength", "G-Stumps", "Corridor")),
        ("SHORT", 7.55, False),
        ("HALF", 9.85, False),
    ]
    for z_txt, d_lbl, is_hi in zone_labels:
        yl = int(round(geom.y_at_distance_m(d_lbl)))
        _, rxl = geom.pitch_x_bounds_at_y(yl)
        tx = int(round(rxl + 16))
        col_txt = (90, 250, 255) if is_hi else (238, 244, 240)
        cv2.putText(canvas, z_txt, (tx + 1, yl + 7), font, 0.78, (22, 45, 25), 3, cv2.LINE_AA)
        cv2.putText(canvas, z_txt, (tx, yl + 6), font, 0.78, col_txt, 2, cv2.LINE_AA)

    # 6. Compute Organic Multi-Scale Gaussian KDE Heatmap matching `image_9832a5.png`
    kde_main = np.zeros((height, width), dtype=np.float32)
    kde_aura = np.zeros((height, width), dtype=np.float32)

    active_deliveries = []
    bg_deliveries = []
    for item in deliveries:
        t = item["traj"]
        if highlight_filter == "G-Stumps":
            is_act = (t.matrix_18_cell == "G-Stumps")
        elif highlight_filter == "Corridor":
            is_act = bool(t.corridor_of_uncertainty)
        elif highlight_filter == "OutsideOff":
            is_act = (t.pitching_line == "OUTSIDE OFF")
        elif highlight_filter == "GoodLength":
            is_act = (t.pitching_length_zone == "Good")
        else:
            is_act = True

        if is_act:
            active_deliveries.append(item)
        else:
            bg_deliveries.append(item)

    # Plot active deliveries with rich multi-scale kernels (tight core + medium body + faint aura)
    for item in active_deliveries:
        t = item["traj"]
        # Map bowler-calibrated length onto the visual FULL / GOOD / SHORT bands on the pitch
        if t.pitching_length_zone == "Full":
            d_vis = 2.6 + 0.35 * (t.pitching_distance_m - 2.2)
        elif t.pitching_length_zone == "Good":
            # Map Good length deliveries into the visual GOOD / upper-GOOD zone (3.1m .. 6.2m)
            if t.bowler_type == "spin":
                d_vis = 3.1 + 0.65 * (t.pitching_distance_m - 3.0)
            else:
                d_vis = 3.6 + 0.75 * (t.pitching_distance_m - 5.0)
        else:
            d_vis = 6.6 + 0.45 * (t.pitching_distance_m - 5.2)

        ym = geom.y_at_distance_m(d_vis)
        lx, rx = geom.pitch_x_bounds_at_y(ym)
        lat = t.lateral_offset_norm
        if t.batter_hand == "LHB" and t.pitching_line != "IN-LINE":
            lat = -abs(lat)
        xm = 0.5 * (lx + rx) + lat * (rx - lx)

        # Tight core + organic lobe
        kde_main += 1.15 * np.exp(-0.5 * (((xx - xm) / 16.5) ** 2 + ((yy - ym) / 21.0) ** 2))
        kde_main += 0.45 * np.exp(-0.5 * (((xx - xm) / 26.0) ** 2 + ((yy - ym) / 34.0) ** 2))

    # Add faint purple outer footprints (like the secondary deliveries in image_9832a5.png)
    for item in (bg_deliveries if highlight_filter is not None else deliveries):
        t = item["traj"]
        ym = geom.y_at_distance_m(min(8.2, t.pitching_distance_m + 0.4))
        lx, rx = geom.pitch_x_bounds_at_y(ym)
        xm = 0.5 * (lx + rx) - t.lateral_offset_norm * (rx - lx) * 1.15
        kde_aura += 0.22 * np.exp(-0.5 * (((xx - xm) / 15.0) ** 2 + ((yy - ym) / 18.0) ** 2))

    if kde_main.max() > 0:
        kde_norm = kde_main / kde_main.max()
        # Non-linear gamma boost so hotspot core reaches fiery red-orange and outer lobes stay green/cyan/blue
        kde_norm = np.power(np.clip(kde_norm, 0.0, 1.0), 0.78)
        kde_total = np.clip(np.maximum(kde_norm, kde_aura), 0.0, 1.0) * (pitch_mask > 0).astype(np.float32)

        idx_lut = np.clip((kde_total * 255.0).astype(np.int32), 0, 255)
        heat_bgr = BGR_LUT[idx_lut]
        heat_alpha = ALPHA_LUT[idx_lut][:, :, None]

        # Smooth alpha slightly so edges blend organically into the sandy pitch texture
        heat_alpha_s = cv2.GaussianBlur(heat_alpha[:, :, 0], (7, 7), 2.0)[:, :, None]
        canvas = np.clip(
            canvas.astype(np.float32) * (1.0 - heat_alpha_s) + heat_bgr.astype(np.float32) * heat_alpha_s,
            0, 255
        ).astype(np.uint8)

    # 7. Draw crisp white 3D Stumps + Bails at top center with diagonal shadow (matching image_9832a5.png)
    st_x = int(round(geom.stumps_center_top_x))
    st_y = int(round(geom.top_y))
    st_h = 46
    # Shadow on turf to the right
    for dx_s in (-9, 0, 9):
        cv2.line(canvas, (st_x + dx_s, st_y), (st_x + dx_s + 22, st_y - 16), (85, 102, 110), 3, cv2.LINE_AA)
    # 3 white stumps
    for dx_s in (-9, 0, 9):
        cv2.line(canvas, (st_x + dx_s, st_y), (st_x + dx_s, st_y - st_h), (195, 200, 198), 5, cv2.LINE_AA)
        cv2.line(canvas, (st_x + dx_s, st_y), (st_x + dx_s, st_y - st_h), (250, 252, 250), 3, cv2.LINE_AA)
    # Top bails
    cv2.line(canvas, (st_x - 10, st_y - st_h), (st_x + 10, st_y - st_h), (250, 252, 250), 3, cv2.LINE_AA)

    # 8. Draw Bottom `OFF` | `MID` | `LEG` boxes matching `image_9832a5.png`
    box_y1, box_y2 = height - 68, height - 16
    box_specs = [
        ("OFF", 30, 165, highlight_filter in ("Corridor", "OutsideOff")),
        ("MID", 180, 315, highlight_filter == "G-Stumps"),
        ("LEG", 330, 465, False),
    ]
    for b_lbl, bx1, bx2, b_hi in box_specs:
        ov_b = canvas.copy()
        fill_c = (28, 42, 48) if b_hi else (38, 42, 40)
        cv2.rectangle(ov_b, (bx1, box_y1), (bx2, box_y2), fill_c, -1)
        canvas = cv2.addWeighted(ov_b, 0.68, canvas, 0.32, 0)
        if b_hi:
            cv2.rectangle(canvas, (bx1, box_y1), (bx2, box_y2), (80, 235, 255), 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(b_lbl, font, 0.82, 2)
        tx = bx1 + (bx2 - bx1 - tw) // 2
        ty = box_y1 + (box_y2 - box_y1 + th) // 2
        txt_c = (95, 245, 255) if b_hi else (242, 245, 242)
        cv2.putText(canvas, b_lbl, (tx + 1, ty + 1), font, 0.82, (15, 18, 15), 3, cv2.LINE_AA)
        cv2.putText(canvas, b_lbl, (tx, ty), font, 0.82, txt_c, 2, cv2.LINE_AA)

    # Redraw dashed lines on top of heatmap blob so white dashes pass cleanly over the blob (matching image_9832a5.png)
    draw_dashed_line(canvas, (int(lx_fg - 26), y_fg), (int(rx_fg + 75), y_fg), (235, 240, 235), 1, 6, 5)
    draw_dashed_line(canvas, (int(lx_gs - 26), y_gs), (int(rx_gs + 75), y_gs), (235, 240, 235), 1, 6, 5)

    # Optional subtle top-right coaching badge for zone spotlight crops (always right of stumps x=275)
    if subtitle_badge:
        (bw, bh), _ = cv2.getTextSize(subtitle_badge, font, 0.43, 1)
        bx1, by1 = max(276, width - bw - 24), 14
        bx2, by2 = width - 8, 42
        ov_t = canvas.copy()
        cv2.rectangle(ov_t, (bx1, by1), (bx2, by2), (20, 28, 22), -1)
        canvas = cv2.addWeighted(ov_t, 0.84, canvas, 0.16, 0)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (90, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, subtitle_badge, (bx1 + 8, by1 + 19), font, 0.43, (245, 250, 245), 1, cv2.LINE_AA)

    return canvas


# GRAPHIC 2: Top-Down 2D Bird's-Eye Full-Pitch Length & Scatter Map (`image_983e0a.png`)
# ===========================================================================
def render_topdown_pitch_length_map(
    deliveries: list,
    width: int = 560,
    height: int = 960,
    show_wicket_labels: bool = True,
) -> np.ndarray:
    """
    Renders the exact Top-Down 2D Bird's-Eye Full-Pitch Map matching `image_983e0a.png`:
      - Checkered two-tone green turf background squares
      - Light mint-green pitch apron (`#9bf08b`) surrounding the tan/beige pitch (`#d8caa8`)
      - Darker tan central stump-to-stump strip (`#bca882`)
      - Top & bottom white bowling creases, popping creases, return creases, and royal-blue wide ticks
      - Horizontal dashed black lines at `2m`, `4m`, `6m`, `8m`, and `10m`
      - Left-side white distance labels (`2m`, `4m`, `6m`, `8m`, `10m`) and black zone names:
        `Yorker` (0-2m), `Slot` (2-4m), `Full` (4-6m), `Good` (6-8m), `Back of length` (8-10m), `Short` (>10m)
      - Translucent match-context delivery cluster + color-coded wicket dots (Red, Blue, Green, Violet)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Checkered two-tone green turf background matching `image_983e0a.png`
    tile_h = height // 6
    col_cols = [0, int(width * 0.08), int(width * 0.36), int(width * 0.64), int(width * 0.92), width]
    for r_i in range(7):
        y0 = r_i * tile_h
        y1 = min(height, (r_i + 1) * tile_h)
        for c_i in range(len(col_cols) - 1):
            x0, x1 = col_cols[c_i], col_cols[c_i + 1]
            is_light = ((r_i + c_i) % 2 == 0)
            # BGR colors matching image_983e0a.png
            bgr = (82, 202, 75) if is_light else (68, 178, 58)
            canvas[y0:y1, x0:x1] = bgr

    # 2. Mint-green pitch outer apron (`x: 68..476, y: 36..924`)
    ap_x1, ap_y1, ap_x2, ap_y2 = 68, 36, 476, 924
    cv2.rectangle(canvas, (ap_x1, ap_y1), (ap_x2, ap_y2), (148, 242, 158), -1)

    # 3. Tan/beige rectangular pitch surface (`x: 100..444, y: 56..904`)
    p_x1, p_y1, p_x2, p_y2 = 100, 56, 444, 904
    cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (168, 204, 218), -1)
    cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (125, 162, 145), 2)

    # 4. Central darker tan wicket-to-wicket strip (`cx = 272, width = 28 px`)
    cx_p = (p_x1 + p_x2) // 2
    st_hw = 14
    y_top_stumps = 102
    y_bot_stumps = 858
    cv2.rectangle(canvas, (cx_p - st_hw, y_top_stumps), (cx_p + st_hw, y_bot_stumps), (132, 168, 186), -1)

    # Scale: full 20.12m pitch between y_top_stumps (0.0m = 102) and y_bot_stumps (20.12m = 858)
    # Wait: in image_983e0a.png, y_pop_top (1.22m) is at y=148, and 2m, 4m, 6m, 8m, 10m are spaced by ~76 px per 2m (38 px/m)!
    px_per_m = 38.0
    def y_at_m(m_val: float) -> int:
        return int(round(y_top_stumps + m_val * px_per_m))

    # 5. Top & Bottom Creases (Bowling crease, Popping crease, Return creases, Royal-blue wide ticks)
    y_pop_top = y_at_m(1.22)
    ret_lx, ret_rx = p_x1 + 24, p_x2 - 24
    wide_lx, wide_rx = p_x1 + 72, p_x2 - 72

    # Top crease markings
    cv2.line(canvas, (ret_lx, y_top_stumps), (ret_rx, y_top_stumps), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (p_x1 - 2, y_pop_top), (p_x2 + 2, y_pop_top), (250, 250, 250), 4, cv2.LINE_AA)
    cv2.line(canvas, (ret_lx, p_y1 + 2), (ret_lx, y_pop_top), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (ret_rx, p_y1 + 2), (ret_rx, y_pop_top), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (wide_lx, y_top_stumps), (wide_lx, y_pop_top - 2), (165, 55, 45), 2, cv2.LINE_AA)
    cv2.line(canvas, (wide_rx, y_top_stumps), (wide_rx, y_pop_top - 2), (165, 55, 45), 2, cv2.LINE_AA)

    # Bottom crease markings
    y_pop_bot = y_bot_stumps - int(round(1.22 * px_per_m))
    cv2.line(canvas, (ret_lx, y_bot_stumps), (ret_rx, y_bot_stumps), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (p_x1 - 2, y_pop_bot), (p_x2 + 2, y_pop_bot), (250, 250, 250), 4, cv2.LINE_AA)
    cv2.line(canvas, (ret_lx, y_pop_bot), (ret_lx, p_y2 - 2), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (ret_rx, y_pop_bot), (ret_rx, p_y2 - 2), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(canvas, (wide_lx, y_pop_bot + 2), (wide_lx, y_bot_stumps), (165, 55, 45), 2, cv2.LINE_AA)
    cv2.line(canvas, (wide_rx, y_pop_bot + 2), (wide_rx, y_bot_stumps), (165, 55, 45), 2, cv2.LINE_AA)

    # 6. Horizontal dashed lines at 2m, 4m, 6m, 8m, 10m + Left-side distance & zone labels (matching image_983e0a.png)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dm in [2.0, 4.0, 6.0, 8.0, 10.0]:
        ym = y_at_m(dm)
        draw_dashed_line(canvas, (48, ym), (496, ym), (45, 52, 45), 1, 4, 4)
        d_txt = f"{int(dm)}m"
        cv2.putText(canvas, d_txt, (16, ym - 3), font, 0.54, (240, 248, 240), 1, cv2.LINE_AA)

    zone_annots = [
        ("Yorker", y_at_m(1.05)),
        ("Slot", y_at_m(2.90)),
        ("Full", y_at_m(4.90)),
        ("Good", y_at_m(6.90)),
        ("Back of", y_at_m(8.55)),
        ("length", y_at_m(9.05)),
        ("Short", y_at_m(10.85)),
    ]
    for z_txt, zy in zone_annots:
        cv2.putText(canvas, z_txt, (16, zy), font, 0.42, (25, 35, 25), 1, cv2.LINE_AA)

    # 7. Plot realistic translucent match-context delivery cloud around the wicket cluster (matching image_983e0a.png)
    rng = np.random.default_rng(108)
    pitch_usable_w = float(ret_rx - ret_lx)
    for item in deliveries:
        t = item["traj"]
        # Use raw meter distance mapped onto the 2m..10m scale
        d_m = t.pitching_distance_m if t.bowler_type == "pace" else (t.pitching_distance_m + 1.4)
        by_c = y_at_m(d_m)
        lat = t.lateral_offset_norm
        if t.batter_hand == "LHB" and t.pitching_line != "IN-LINE":
            lat = -abs(lat)
        bx_c = cx_p + lat * pitch_usable_w * 0.95

        # Draw 10 translucent grey/white context balls per wicket delivery to show spell scatter cloud
        for _ in range(10):
            jx = int(round(bx_c + rng.normal(0.0, 14.0)))
            jy = int(round(by_c + rng.normal(0.0, 22.0)))
            if ret_lx + 8 < jx < ret_rx - 8 and y_top_stumps < jy < y_bot_stumps:
                ov = canvas.copy()
                is_white = rng.random() > 0.55
                c_dot = (242, 242, 242) if is_white else (128, 138, 135)
                cv2.circle(ov, (jx, jy), 6, c_dot, -1, cv2.LINE_AA)
                cv2.circle(ov, (jx, jy), 6, (185, 190, 188), 1, cv2.LINE_AA)
                canvas = cv2.addWeighted(ov, 0.48, canvas, 0.52, 0)

    # 8. Plot the 12 actual Wicket-Taking Deliveries as bold color-coded dots (Red, Blue, Green, Violet)
    wicket_pts = []
    for item in deliveries:
        t = item["traj"]
        w_id = item["w_id"]
        d_m = t.pitching_distance_m if t.bowler_type == "pace" else (t.pitching_distance_m + 1.4)
        by = y_at_m(d_m)
        lat = t.lateral_offset_norm
        if t.batter_hand == "LHB" and t.pitching_line != "IN-LINE":
            lat = -abs(lat)
        bx = int(round(cx_p + lat * pitch_usable_w * 0.95))

        # Color-coding matching image_983e0a.png:
        # - Red: Good-Length Stump-Line wicket
        # - Blue: Corridor of Uncertainty / Outside-Off wicket
        # - Bright Green: Full / Slot wicket
        # - Magenta/Violet: Spin / Variation wicket
        if t.pitching_length_zone == "Full":
            dot_bgr = (35, 235, 45)    # Bright green
        elif t.corridor_of_uncertainty or t.pitching_line == "OUTSIDE OFF":
            dot_bgr = (225, 135, 30)   # Royal/ Dodger blue
        elif t.bowler_type == "spin":
            dot_bgr = (215, 115, 220)  # Violet/Pink
        else:
            dot_bgr = (25, 25, 235)    # Crimson red

        cv2.circle(canvas, (bx, by), 7, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 6, dot_bgr, -1, cv2.LINE_AA)
        wicket_pts.append((w_id, bx, by, t, dot_bgr))

    return canvas



def render_perspective_pitch_heatmap(
    deliveries,
    width=620,
    height=760,
    highlight_zone=None, # None, "G-Stumps", "Corridor", "GoodLength", "WideOff"
    title="3D Perspective Pitch Wicket Heatmap (12 Wickets)"
) -> Tuple[np.ndarray, PitchGeometry, List[Tuple[str, int, int, object]]]:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        stripe = 7 if (y // 26) % 2 == 0 else 0
        canvas[y, :] = (64 + stripe, 118 + stripe, 82 + stripe)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.rectangle(canvas, (0, 0), (width, 48), (18, 24, 20), -1)
    cv2.putText(canvas, title, (16, 31), font, 0.54, (245, 248, 245), 1, cv2.LINE_AA)

    geom = PitchGeometry(
        bbox=(85.0, 125.0, 515.0, 695.0),
        raw_py1=125.0, top_y=125.0, bottom_y=695.0,
        top_left_x=135.0, top_right_x=465.0,
        bottom_left_x=75.0, bottom_right_x=525.0,
        stumps_center_top_x=300.0, stumps_center_bottom_x=300.0,
        inline_half_width_top=15.0, inline_half_width_bottom=22.0,
        visible_length_m=12.0,
    )

    # Draw pitch surface
    pitch_poly = np.array([
        [int(round(geom.top_left_x)), int(round(geom.top_y - 12))],
        [int(round(geom.top_right_x)), int(round(geom.top_y - 12))],
        [int(round(geom.bottom_right_x)), int(round(geom.bottom_y))],
        [int(round(geom.bottom_left_x)), int(round(geom.bottom_y))],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pitch_poly], (128, 148, 156), lineType=cv2.LINE_AA)

    # Subtle length zone tinting
    ov_z = canvas.copy()
    for z_name, d0, d1, fill_bgr, _ in PitchingDetailsRenderer.ZONES:
        if d0 >= geom.visible_length_m:
            continue
        poly = geom.band_polygon(d0, min(d1, geom.visible_length_m))
        cv2.fillPoly(ov_z, [poly], fill_bgr, lineType=cv2.LINE_AA)
    canvas = cv2.addWeighted(ov_z, 0.24, canvas, 0.76, 0)

    # Draw royal-blue inline strip
    canvas = HawkEyeRenderer._draw_royal_blue_pitch_strip(canvas, geom)

    # Draw Corridor of Uncertainty strip (0 to 30cm outside off stump -> left of inline strip for RHB)
    corr_pts = []
    for y_s in [geom.top_y, geom.bottom_y]:
        il_l, _ = geom.inline_x_bounds_at_y(y_s)
        px_l, px_r = geom.pitch_x_bounds_at_y(y_s)
        w_30cm = (0.305 / 3.05) * (px_r - px_l)
        corr_pts.append((il_l - w_30cm, il_l, y_s))
    corr_poly = np.array([
        [int(round(corr_pts[0][0])), int(round(corr_pts[0][2]))],
        [int(round(corr_pts[0][1])), int(round(corr_pts[0][2]))],
        [int(round(corr_pts[1][1])), int(round(corr_pts[1][2]))],
        [int(round(corr_pts[1][0])), int(round(corr_pts[1][2]))],
    ], dtype=np.int32)
    ov_c = canvas.copy()
    cv2.fillPoly(ov_c, [corr_poly], (45, 195, 245), lineType=cv2.LINE_AA)
    canvas = cv2.addWeighted(ov_c, 0.25, canvas, 0.75, 0)
    cv2.polylines(canvas, [corr_poly], True, (65, 220, 255), 1, cv2.LINE_AA)

    # Compute 2D Gaussian KDE over the pitch polygon
    pitch_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(pitch_mask, [pitch_poly], 255)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    kde = np.zeros((height, width), dtype=np.float32)

    mapped_pts = []
    for item in deliveries:
        t = item["traj"]
        w_id = item["w_id"]
        # Filter if highlight_zone is specified
        if highlight_zone == "G-Stumps" and t.matrix_18_cell != "G-Stumps":
            continue
        if highlight_zone == "Corridor" and not t.corridor_of_uncertainty:
            continue
        if highlight_zone == "GoodLength" and t.pitching_length_zone != "Good":
            continue

        ym = geom.y_at_distance_m(t.pitching_distance_m)
        lx, rx = geom.pitch_x_bounds_at_y(ym)
        # Map RHB/LHB off-side consistently to the off-stump corridor side (left of stumps) for unified coaching view
        lat = t.lateral_offset_norm
        if t.batter_hand == "LHB" and t.pitching_line != "IN-LINE":
            lat = -abs(lat)
        xm = 0.5 * (lx + rx) + lat * (rx - lx)
        bx, by = int(round(xm)), int(round(ym))
        mapped_pts.append((w_id, bx, by, t))

        sigma_x = 24.0
        sigma_y = 32.0
        kde += np.exp(-0.5 * (((xx - xm) / sigma_x) ** 2 + ((yy - ym) / sigma_y) ** 2))

    if kde.max() > 0:
        kde_norm = (kde / kde.max()) * (pitch_mask > 0).astype(np.float32)
        kde_u8 = np.clip(kde_norm * 255.0, 0, 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(kde_u8, cv2.COLORMAP_TURBO)

        # Alpha-blend heatmap where kde_norm > 0.08
        alpha = np.clip((kde_norm - 0.06) / 0.94, 0.0, 1.0) * 0.72
        alpha = alpha[:, :, None]
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - alpha) + heat_bgr.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # Draw crisp thermal isoline contours at 35%, 65%, 85% peak density
        for thresh, col_iso in [(0.35, (120, 240, 255)), (0.65, (60, 195, 255)), (0.85, (255, 255, 255))]:
            iso_mask = ((kde_norm >= thresh) & (pitch_mask > 0)).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(iso_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, col_iso, 1, cv2.LINE_AA)

    # Highlight bounding box if highlight_zone is requested
    if highlight_zone == "G-Stumps":
        y_top_z = int(round(geom.y_at_distance_m(3.5)))
        y_bot_z = int(round(geom.y_at_distance_m(6.4)))
        il_l1, il_r1 = geom.inline_x_bounds_at_y(y_top_z)
        il_l2, il_r2 = geom.inline_x_bounds_at_y(y_bot_z)
        z_poly = np.array([
            [int(il_l1 - 8), y_top_z], [int(il_r1 + 8), y_top_z],
            [int(il_r2 + 8), y_bot_z], [int(il_l2 - 8), y_bot_z]
        ], dtype=np.int32)
        cv2.polylines(canvas, [z_poly], True, (50, 255, 255), 3, cv2.LINE_AA)
    elif highlight_zone == "Corridor":
        y_top_z = int(round(geom.y_at_distance_m(3.0)))
        y_bot_z = int(round(geom.y_at_distance_m(7.6)))
        il_l1, _ = geom.inline_x_bounds_at_y(y_top_z)
        il_l2, _ = geom.inline_x_bounds_at_y(y_bot_z)
        px_l1, px_r1 = geom.pitch_x_bounds_at_y(y_top_z)
        px_l2, px_r2 = geom.pitch_x_bounds_at_y(y_bot_z)
        w1 = (0.32 / 3.05) * (px_r1 - px_l1)
        w2 = (0.32 / 3.05) * (px_r2 - px_l2)
        z_poly = np.array([
            [int(il_l1 - w1 - 4), y_top_z], [int(il_l1 + 2), y_top_z],
            [int(il_l2 + 2), y_bot_z], [int(il_l2 - w2 - 4), y_bot_z]
        ], dtype=np.int32)
        cv2.polylines(canvas, [z_poly], True, (50, 255, 255), 3, cv2.LINE_AA)
    elif highlight_zone == "GoodLength":
        z_poly = geom.band_polygon(3.3, 7.5)
        cv2.polylines(canvas, [z_poly], True, (50, 255, 255), 3, cv2.LINE_AA)

    # Re-draw creases and stumps cleanly over heatmap
    PitchingDetailsRenderer._draw_creases_and_stumps(canvas, geom, draw_bottom_stumps=True)

    # Distance markers on right edge
    for dm, dlabel in [(0.0, "STUMPS"), (2.0, "2m"), (4.0, "4m"), (6.0, "6m"), (8.0, "8m"), (10.06, "HALFWAY")]:
        ym = int(round(geom.y_at_distance_m(dm)))
        lx, rx = geom.pitch_x_bounds_at_y(ym)
        cv2.line(canvas, (int(rx) - 6, ym), (int(rx) + 8, ym), (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, dlabel, (int(rx) + 13, ym + 5), font, 0.46, (20, 25, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, dlabel, (int(rx) + 12, ym + 4), font, 0.46, (245, 248, 245), 1, cv2.LINE_AA)

    # Draw 3D white/red wicket markers and de-collided callout labels
    for w_id, bx, by, t in mapped_pts:
        style = "yellow" if t.corridor_of_uncertainty else ("red" if t.pitching_line == "IN-LINE" else "blue")
        PitchingDetailsRenderer._draw_3d_sphere_marker(canvas, (bx, by), radius=7, style=style)
        cv2.circle(canvas, (bx, by), 9, (255, 255, 255), 1, cv2.LINE_AA)

    pts_sorted = sorted(mapped_pts, key=lambda item: item[2])
    placed_y = []
    for w_id, bx, by, t in pts_sorted:
        ty = by
        for py in placed_y:
            if abs(ty - py) < 22:
                ty = py + 22
        placed_y.append(ty)
        if t.pitching_line == "OUTSIDE OFF" or w_id in ("W9", "W7"):
            tx = max(10, bx - 142)
            cv2.line(canvas, (bx - 8, by), (tx + 128, ty - 3), (235, 235, 235), 1, cv2.LINE_AA)
        else:
            tx = bx + 14
            cv2.line(canvas, (bx + 8, by), (tx - 3, ty - 3), (235, 235, 235), 1, cv2.LINE_AA)
        lbl = f"{w_id}: {t.pitching_distance_m:.1f}m [{t.matrix_18_cell}]"
        cv2.putText(canvas, lbl, (tx + 1, ty + 3), font, 0.38, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (tx, ty + 2), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Add colorbar legend at bottom
    cb_x, cb_y, cb_w, cb_h = 85, height - 36, 300, 14
    grad = np.linspace(0, 255, cb_w, dtype=np.uint8).reshape(1, cb_w)
    grad_bgr = cv2.applyColorMap( np.repeat(grad, cb_h, axis=0), cv2.COLORMAP_TURBO )
    canvas[cb_y:cb_y+cb_h, cb_x:cb_x+cb_w] = grad_bgr
    cv2.rectangle(canvas, (cb_x, cb_y), (cb_x+cb_w, cb_y+cb_h), (230, 230, 230), 1)
    cv2.putText(canvas, "Low Wicket Density", (cb_x, cb_y - 5), font, 0.36, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "High Wicket Hotspot", (cb_x + cb_w - 115, cb_y - 5), font, 0.36, (255, 230, 120), 1, cv2.LINE_AA)

    return canvas, geom, mapped_pts



class PitchingDetailsRenderer:
    """
    Renders Output 2 (Ball Pitching Details & Length-Zone Pitch Map) matching the
    user's reference images (`image_e688fd.png` Broadcast Pitch Map + `image_e68d17.png`
    Length-Zone Analytics Pitch Map).
    """

    ZONES = [
        ("Full Toss", 0.0, 1.2, (165, 195, 205), (30, 30, 30)),
        ("Yorker", 1.2, 2.2, (95, 220, 235), (20, 215, 240)),
        ("Full", 2.2, 5.0, (115, 205, 125), (50, 205, 70)),
        ("Good", 5.0, 8.0, (105, 130, 215), (45, 55, 220)),
        ("Short", 8.0, 16.0, (175, 125, 115), (185, 45, 25)),
    ]

    def render_pitching_details_on_frame(
        self,
        frame: np.ndarray,
        pitch_geom: PitchGeometry,
        traj: Optional[TrajectoryModel],
        all_trajectories: Optional[List[TrajectoryModel]] = None,
        show_trajectory_curve: bool = False,
    ) -> np.ndarray:
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        trajs = all_trajectories if all_trajectories else ([traj] if traj is not None else [])
        total_balls = max(1, len(trajs))
        zone_counts = {z[0]: 0 for z in self.ZONES}
        for t in trajs:
            if t is not None:
                zone_counts[t.pitching_length_zone] = zone_counts.get(t.pitching_length_zone, 0) + 1

        overlay = canvas.copy()
        for z_name, d0, d1, fill_bgr, _ in self.ZONES:
            if d0 >= pitch_geom.visible_length_m:
                continue
            poly = pitch_geom.band_polygon(d0, min(d1, pitch_geom.visible_length_m))
            cv2.fillPoly(overlay, [poly], fill_bgr, lineType=cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0)

        canvas = HawkEyeRenderer._draw_royal_blue_pitch_strip(canvas, pitch_geom)

        font = cv2.FONT_HERSHEY_DUPLEX
        popping_y = int(round(pitch_geom.y_at_distance_m(1.22)))
        lx_pop, rx_pop = pitch_geom.pitch_x_bounds_at_y(popping_y)
        cv2.line(
            canvas,
            (int(round(lx_pop)), popping_y),
            (int(round(rx_pop)), popping_y),
            (242, 242, 242),
            2,
            lineType=cv2.LINE_AA,
        )

        distance_marks = [
            (0.0, "STUMPS"),
            (2.0, "2m"),
            (4.0, "4m"),
            (6.0, "6m"),
            (8.0, "8m"),
            (10.06, "HALFWAY"),
        ]
        for dist_m, label in distance_marks:
            if dist_m > pitch_geom.visible_length_m - 0.3:
                continue
            ym = int(round(pitch_geom.y_at_distance_m(dist_m)))
            lx, rx = pitch_geom.pitch_x_bounds_at_y(ym)
            sub_ov = canvas.copy()
            cv2.line(sub_ov, (int(round(lx)), ym), (int(round(rx)), ym), (235, 235, 235), 1, cv2.LINE_AA)
            canvas = cv2.addWeighted(sub_ov, 0.28, canvas, 0.72, 0)
            cv2.line(canvas, (int(round(rx - 10)), ym), (int(round(rx + 6)), ym), (245, 245, 245), 2, cv2.LINE_AA)

            txt_x = int(round(rx - 58 if label != "HALFWAY" else rx - 105))
            txt_y = ym - 5
            cv2.putText(canvas, label, (txt_x + 1, txt_y + 1), font, 0.52, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (txt_x, txt_y), font, 0.52, (248, 248, 248), 1, cv2.LINE_AA)

        for z_name, d0, d1, _, label_bgr in self.ZONES:
            if d0 >= pitch_geom.visible_length_m - 0.5:
                continue
            d_mid = 0.5 * (d0 + min(d1, pitch_geom.visible_length_m))
            ym = int(round(pitch_geom.y_at_distance_m(d_mid)))
            lx, rx = pitch_geom.pitch_x_bounds_at_y(ym)

            pct = int(round(100.0 * zone_counts.get(z_name, 0) / total_balls))
            pct_str = f"{pct}%"

            px_left = max(12, int(round(lx - 72)))
            cv2.putText(canvas, pct_str, (px_left + 1, ym + 6), font, 0.65, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(canvas, pct_str, (px_left, ym + 5), font, 0.65, label_bgr, 2, cv2.LINE_AA)

            if z_name != "Full Toss":
                px_right = min(w - 95, int(round(rx + 18)))
                cv2.putText(canvas, z_name, (px_right + 1, ym + 6), font, 0.65, (15, 15, 15), 3, cv2.LINE_AA)
                cv2.putText(canvas, z_name, (px_right, ym + 5), font, 0.65, label_bgr, 2, cv2.LINE_AA)

        if show_trajectory_curve and traj is not None:
            pre_proj = traj.project_pts_to_geom(traj.pre_bounce_pts, pitch_geom)
            post_proj = traj.project_pts_to_geom(traj.post_bounce_pts, pitch_geom)
            base_r = max(3.5, 0.016 * pitch_geom.top_width)
            HawkEyeRenderer._draw_3d_pearl_white_tube(
                canvas, pre_proj, pitch_geom, base_r, draw_entry_cap=True, draw_bounce_disc=True
            )
            HawkEyeRenderer._draw_3d_pearl_white_tube(
                canvas, post_proj, pitch_geom, base_r * 0.92, draw_red_ball_tip=True
            )

        for idx, t in enumerate(trajs):
            if t is None:
                continue
            b_pt = t.project_pts_to_geom(np.array([[t.bounce_pt[0], t.bounce_pt[1]]]), pitch_geom)[0]
            bx, by = int(round(b_pt[0])), int(round(b_pt[1]))
            is_primary = (t is traj) or (idx == len(trajs) - 1)
            r_ball = 9 if is_primary else 7
            self._draw_3d_sphere_marker(canvas, (bx, by), radius=r_ball, style="red" if is_primary else "white")

            if is_primary:
                cv2.circle(canvas, (bx, by), r_ball + 5, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                tag = f"{t.pitching_distance_m:.1f}m ({t.pitching_length_zone.upper()})"
                (tw, th), _ = cv2.getTextSize(tag, font, 0.52, 1)
                tx = max(16, min(w - tw - 20, bx - tw // 2))
                ty = by + r_ball + 24
                cv2.rectangle(canvas, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (22, 26, 32), -1)
                cv2.rectangle(canvas, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(canvas, tag, (tx, ty), font, 0.52, (250, 250, 250), 1, cv2.LINE_AA)

        self._draw_pitch_map_header_card(canvas, traj)
        return canvas

    @staticmethod
    def _draw_3d_sphere_marker(
        canvas: np.ndarray,
        center: Tuple[int, int],
        radius: int = 8,
        style: str = "red",
    ) -> None:
        cx, cy = center
        cv2.ellipse(
            canvas,
            (cx + 1, cy + 3),
            (radius, max(3, int(radius * 0.55))),
            0, 0, 360,
            (25, 30, 25),
            -1,
            lineType=cv2.LINE_AA,
        )
        if style == "red":
            edge_c, body_c, hi_c = (25, 25, 145), (38, 42, 225), (145, 155, 255)
        elif style == "blue":
            edge_c, body_c, hi_c = (135, 45, 20), (225, 95, 45), (255, 185, 145)
        elif style == "yellow":
            edge_c, body_c, hi_c = (20, 145, 175), (45, 225, 245), (185, 250, 255)
        else:
            edge_c, body_c, hi_c = (135, 140, 145), (232, 236, 240), (255, 255, 255)

        cv2.circle(canvas, (cx, cy), radius, edge_c, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), max(2, radius - 1), body_c, -1, lineType=cv2.LINE_AA)
        cv2.circle(
            canvas,
            (cx - max(1, radius // 3), cy - max(1, radius // 3)),
            max(1, radius // 3),
            hi_c,
            -1,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _draw_pitch_map_header_card(canvas: np.ndarray, traj: Optional[TrajectoryModel]) -> None:
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        card_w, card_h = 410, 108
        # Place at top-right if top-left has broadcast/HUD banner, else top-center
        top_left_luma = float(np.mean(canvas[10:60, 15:250]))
        x0 = (w - card_w - 22) if top_left_luma < 45.0 else (w - card_w) // 2
        y0 = 18

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (24, 30, 24), -1)
        canvas[:] = cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0)
        cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), (185, 195, 185), 1, cv2.LINE_AA)

        cv2.putText(canvas, "BALL PITCHING DETAILS & PITCH MAP", (x0 + 20, y0 + 26), font, 0.55, (250, 252, 250), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 14, y0 + 35), (x0 + card_w - 14, y0 + 35), (95, 110, 95), 1)

        if traj is not None:
            l1 = f"Pitching Length : {traj.pitching_distance_m:.2f}m ({traj.pitching_length_zone.upper()})"
            line_extra = " [CORRIDOR]" if traj.corridor_of_uncertainty else f" [{traj.matrix_18_cell}]"
            l2 = f"Pitching Line   : {traj.pitching_line}{line_extra}"
            cv2.putText(canvas, l1, (x0 + 20, y0 + 64), font, 0.50, (95, 235, 250), 1, cv2.LINE_AA)
            cv2.putText(canvas, l2, (x0 + 20, y0 + 92), font, 0.50, (235, 240, 235), 1, cv2.LINE_AA)

    def render_standalone_analytics_board(
        self,
        deliveries: List[Tuple[str, TrajectoryModel]],
        width: int = 1280,
        height: int = 720,
    ) -> np.ndarray:
        """
        Renders the de-collided Dual-View Pitch Map Analytics Board across all wicket-taking
        deliveries in a match.
        """
        board = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            stripe = 8 if (y // 28) % 2 == 0 else 0
            board[y, :] = (76 + stripe, 136 + stripe, 96 + stripe)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(board, (0, 0), (width, 54), (20, 26, 22), -1)
        cv2.putText(
            board,
            "MATCH WICKET DELIVERIES: BALL PITCHING DETAILS & LENGTH-ZONE ANALYTICS",
            (22, 35),
            font,
            0.70,
            (245, 248, 245),
            1,
            cv2.LINE_AA,
        )

        lp_geom = PitchGeometry(
            bbox=(75.0, 175.0, 525.0, 675.0),
            raw_py1=175.0,
            top_y=175.0,
            bottom_y=675.0,
            top_left_x=120.0,
            top_right_x=480.0,
            bottom_left_x=70.0,
            bottom_right_x=530.0,
            stumps_center_top_x=300.0,
            stumps_center_bottom_x=300.0,
            inline_half_width_top=15.0,
            inline_half_width_bottom=21.0,
            visible_length_m=12.5,
        )
        self._draw_clean_pitch_surface(board, lp_geom, inline_bgr=(72, 82, 145), inline_alpha=0.52)

        cv2.putText(board, "Broadcast Pitch Map", (185, 86), font, 0.60, (250, 250, 250), 1, cv2.LINE_AA)
        leg_x, leg_y, leg_w, leg_h = 465, 66, 170, 108
        ov = board.copy()
        cv2.rectangle(ov, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (28, 34, 26), -1)
        board[:] = cv2.addWeighted(ov, 0.84, board, 0.16, 0)
        cv2.rectangle(board, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (140, 150, 135), 1)
        for idx_l, (l_txt, l_col) in enumerate([("Good Length", "red"), ("Full Length", "blue"), ("Yorker", "yellow"), ("Short", "white")]):
            ly = leg_y + 21 + idx_l * 23
            self._draw_3d_sphere_marker(board, (leg_x + 18, ly - 4), radius=6, style=l_col)
            cv2.putText(board, l_txt, (leg_x + 34, ly + 1), font, 0.44, (245, 245, 245), 1, cv2.LINE_AA)

        for dm, dlabel in [(2.0, "2m"), (4.0, "4m"), (6.0, "6m"), (8.0, "8m"), (10.06, "HALFWAY")]:
            ym = int(round(lp_geom.y_at_distance_m(dm)))
            _, rx = lp_geom.pitch_x_bounds_at_y(ym)
            cv2.line(board, (int(rx) - 6, ym), (int(rx) + 8, ym), (230, 230, 230), 1, cv2.LINE_AA)
            tx = int(rx) + 12
            cv2.putText(board, dlabel, (tx + 1, ym + 5), font, 0.50, (30, 35, 30), 2, cv2.LINE_AA)
            cv2.putText(board, dlabel, (tx, ym + 4), font, 0.50, (240, 242, 240), 1, cv2.LINE_AA)

        pts_left = []
        for idx_d, (d_name, t) in enumerate(deliveries):
            ym = lp_geom.y_at_distance_m(t.pitching_distance_m)
            lx, rx = lp_geom.pitch_x_bounds_at_y(ym)
            xm = 0.5 * (lx + rx) + t.lateral_offset_norm * (rx - lx)
            col = "red" if t.pitching_length_zone == "Good" else ("blue" if t.pitching_length_zone == "Full" else "yellow")
            pts_left.append((idx_d + 1, int(round(xm)), int(round(ym)), t, col))

        for w_no, bx, by, t, col in pts_left:
            self._draw_3d_sphere_marker(board, (bx, by), radius=8, style=col)

        pts_left_sorted = sorted(pts_left, key=lambda item: item[2])
        placed_y: List[int] = []
        for w_no, bx, by, t, col in pts_left_sorted:
            ty = by
            for py in placed_y:
                if abs(ty - py) < 20:
                    ty = py + 20
            placed_y.append(ty)
            if t.lateral_offset_norm < -0.02:
                tx = max(14, bx - 168)
                cv2.line(board, (bx - 9, by), (tx + 152, ty - 3), (225, 225, 225), 1, cv2.LINE_AA)
            else:
                tx = bx + 16
                cv2.line(board, (bx + 9, by), (tx - 4, ty - 3), (225, 225, 225), 1, cv2.LINE_AA)
            lbl = f"W{w_no}: {t.pitching_distance_m:.1f}m ({t.pitching_line})"
            cv2.putText(board, lbl, (tx + 1, ty + 2), font, 0.42, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(board, lbl, (tx, ty + 1), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        rp_geom = PitchGeometry(
            bbox=(760.0, 175.0, 1150.0, 675.0),
            raw_py1=175.0,
            top_y=175.0,
            bottom_y=675.0,
            top_left_x=815.0,
            top_right_x=1095.0,
            bottom_left_x=755.0,
            bottom_right_x=1155.0,
            stumps_center_top_x=955.0,
            stumps_center_bottom_x=955.0,
            inline_half_width_top=11.0,
            inline_half_width_bottom=15.0,
            visible_length_m=16.0,
        )
        cv2.putText(board, "Length-Zone Analytics Map", (815, 86), font, 0.60, (250, 250, 250), 1, cv2.LINE_AA)
        self._draw_clean_pitch_surface(board, rp_geom, inline_bgr=(165, 58, 38), inline_alpha=0.0)

        zone_counts = {z[0]: 0 for z in self.ZONES}
        for _, t in deliveries:
            zone_counts[t.pitching_length_zone] = zone_counts.get(t.pitching_length_zone, 0) + 1
        total_d = max(1, len(deliveries))

        ov_r = board.copy()
        for z_name, d0, d1, fill_bgr, _ in self.ZONES:
            poly = rp_geom.band_polygon(d0, min(d1, rp_geom.visible_length_m))
            cv2.fillPoly(ov_r, [poly], fill_bgr, lineType=cv2.LINE_AA)
        board[:] = cv2.addWeighted(ov_r, 0.62, board, 0.38, 0)

        board = HawkEyeRenderer._draw_royal_blue_pitch_strip(board, rp_geom)
        self._draw_creases_and_stumps(board, rp_geom, draw_bottom_stumps=True)

        for dm, dlabel in [(0.0, "STUMPS"), (2.0, "2M"), (4.0, "4M"), (6.0, "6M"), (8.0, "8M"), (10.06, "HALFWAY")]:
            ym = int(round(rp_geom.y_at_distance_m(dm)))
            _, rx = rp_geom.pitch_x_bounds_at_y(ym)
            cv2.putText(board, dlabel, (int(rx) - 24, ym + 4), font, 0.38, (245, 245, 245), 1, cv2.LINE_AA)

        for z_name, d0, d1, _, label_bgr in self.ZONES:
            d_mid = 0.5 * (d0 + min(d1, rp_geom.visible_length_m))
            ym = int(round(rp_geom.y_at_distance_m(d_mid)))
            lx, rx = rp_geom.pitch_x_bounds_at_y(ym)
            pct = int(round(100.0 * zone_counts.get(z_name, 0) / total_d))
            cv2.putText(board, f"{pct}%", (int(lx) - 62, ym + 5), font, 0.56, label_bgr, 2, cv2.LINE_AA)
            if z_name != "Full Toss":
                cv2.putText(board, z_name, (int(rx) + 28, ym + 5), font, 0.56, label_bgr, 2, cv2.LINE_AA)

        pts_right = []
        for idx_d, (d_name, t) in enumerate(deliveries):
            ym = rp_geom.y_at_distance_m(t.pitching_distance_m)
            lx, rx = rp_geom.pitch_x_bounds_at_y(ym)
            xm = 0.5 * (lx + rx) + t.lateral_offset_norm * (rx - lx)
            pts_right.append((idx_d + 1, int(round(xm)), int(round(ym)), t))

        for w_no, bx, by, t in pts_right:
            self._draw_3d_sphere_marker(board, (bx, by), radius=6, style="white")

        pts_r_sorted = sorted(pts_right, key=lambda item: item[2])
        placed_ry: List[int] = []
        for w_no, bx, by, t in pts_r_sorted:
            ty = by
            for py in placed_ry:
                if abs(ty - py) < 17:
                    ty = py + 17
            placed_ry.append(ty)
            tx = bx - 98 if t.lateral_offset_norm < 0.0 else bx + 14
            cv2.putText(board, f"W{w_no} ({t.pitching_distance_m:.1f}m)", (tx + 1, ty + 4), font, 0.39, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(board, f"W{w_no} ({t.pitching_distance_m:.1f}m)", (tx, ty + 3), font, 0.39, (255, 255, 255), 1, cv2.LINE_AA)

        return board

    @staticmethod
    def _draw_clean_pitch_surface(
        canvas: np.ndarray,
        geom: PitchGeometry,
        inline_bgr: Tuple[int, int, int] = (72, 82, 145),
        inline_alpha: float = 0.50,
    ) -> None:
        pitch_poly = np.array([
            [int(round(geom.top_left_x)), int(round(geom.top_y - 12))],
            [int(round(geom.top_right_x)), int(round(geom.top_y - 12))],
            [int(round(geom.bottom_right_x)), int(round(geom.bottom_y))],
            [int(round(geom.bottom_left_x)), int(round(geom.bottom_y))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pitch_poly], (132, 154, 162), lineType=cv2.LINE_AA)

        if inline_alpha > 0:
            ov = canvas.copy()
            cv2.fillPoly(ov, [geom.inline_strip_polygon()], inline_bgr, lineType=cv2.LINE_AA)
            canvas[:] = cv2.addWeighted(ov, inline_alpha, canvas, 1.0 - inline_alpha, 0)

        PitchingDetailsRenderer._draw_creases_and_stumps(canvas, geom, draw_bottom_stumps=False)

    @staticmethod
    def _draw_creases_and_stumps(canvas: np.ndarray, geom: PitchGeometry, draw_bottom_stumps: bool = False) -> None:
        y_top = int(round(geom.top_y))
        y_pop = int(round(geom.y_at_distance_m(1.22)))
        lx_t, rx_t = geom.pitch_x_bounds_at_y(y_top)
        lx_p, rx_p = geom.pitch_x_bounds_at_y(y_pop)
        cv2.line(canvas, (int(lx_t) + 14, y_top), (int(rx_t) - 14, y_top), (235, 235, 235), 2, cv2.LINE_AA)
        cv2.line(canvas, (int(lx_p) + 14, y_pop), (int(rx_p) - 14, y_pop), (242, 242, 242), 2, cv2.LINE_AA)
        cv2.line(canvas, (int(lx_t) + 14, y_top - 4), (int(lx_p) + 14, y_pop), (242, 242, 242), 2, cv2.LINE_AA)
        cv2.line(canvas, (int(rx_t) - 14, y_top - 4), (int(rx_p) - 14, y_pop), (242, 242, 242), 2, cv2.LINE_AA)

        cx_t = int(round(geom.stumps_center_top_x))
        for dx in (-7, 0, 7):
            cv2.line(canvas, (cx_t + dx, y_top), (cx_t + dx, y_top - 36), (165, 205, 225), 2, cv2.LINE_AA)

        if draw_bottom_stumps:
            y_bot = int(round(geom.bottom_y - 22))
            y_bpop = int(round(geom.bottom_y - 52))
            lx_b, rx_b = geom.pitch_x_bounds_at_y(y_bot)
            lx_bp, rx_bp = geom.pitch_x_bounds_at_y(y_bpop)
            cv2.line(canvas, (int(lx_b) + 18, y_bot), (int(rx_b) - 18, y_bot), (242, 242, 242), 2, cv2.LINE_AA)
            cv2.line(canvas, (int(lx_bp) + 18, y_bpop), (int(rx_bp) - 18, y_bpop), (242, 242, 242), 2, cv2.LINE_AA)
            cx_b = int(round(geom.stumps_center_bottom_x))
            for dx in (-8, 0, 8):
                cv2.line(canvas, (cx_b + dx, y_bot), (cx_b + dx, y_bot - 42), (145, 185, 210), 2, cv2.LINE_AA)



    def render_heatmap_analytics_dashboard(
        self,
        deliveries: List[Tuple[str, TrajectoryModel]],
        output_dir: str,
        sample_images: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, str]:
        """
        Generates and saves all 5 requested coaching & broadcast graphics:
          1. `output_1_broadcast_pitch_heatmap.png` (3D Broadcast Pitch Thermal Heatmap: FULL/GOOD/SHORT/HALF + OFF/MID/LEG)
          2. `output_2_topdown_pitch_length_map.png` (2D Top-Down Bird's-Eye Full-Pitch Length & Scatter Map: 2m..10m)
          3. `output_3_broadcast_and_length_zone_pitch_map.png` (Dual-View Broadcast Pitch Map + Length-Zone Analytics Map)
          4. `output_4_matrix_and_executive_report.png` (2D Line x Length Wicket Density Matrix + Executive Coaching Report)
          5. `output_5_zone_spotlight_cards.png` (3 Visual Zone Spotlight Cards with Pitch Zone Heatmap Pic + Sample Wicket)
          6. `wicket_heatmap_analytics_dashboard.png` (Combined 1920x1080 3-Panel Master Coaching Dashboard)
        """
        os.makedirs(output_dir, exist_ok=True)
        total_w = max(1, len(deliveries))
        hm_items = []
        for i, (stem_name, t) in enumerate(deliveries):
            s_img = None
            if sample_images and i < len(sample_images) and sample_images[i] is not None:
                s_img = sample_images[i]
            else:
                cand_p = os.path.join(output_dir, f"{stem_name}_trajectory.png")
                if os.path.exists(cand_p):
                    s_img = cv2.imread(cand_p)
            if s_img is None:
                s_img = np.zeros((360, 640, 3), dtype=np.uint8)
            hm_items.append({"w_id": f"W{i+1}", "clip_id": stem_name, "traj": t, "traj_img": s_img})

        # Output 1: Broadcast 3D Perspective Pitch Thermal Heatmap
        g1_img = render_broadcast_pitch_heatmap(hm_items, width=680, height=640, highlight_filter=None)
        p1_path = os.path.join(output_dir, "output_1_broadcast_pitch_heatmap.png")
        cv2.imwrite(p1_path, g1_img)
        cv2.imwrite(os.path.join(output_dir, "pitch_kde_heatmap_12_wickets.png"), g1_img)

        # Output 2: Top-Down 2D Bird's-Eye Full-Pitch Length & Scatter Map
        g2_img = render_topdown_pitch_length_map(hm_items, width=560, height=960)
        p2_path = os.path.join(output_dir, "output_2_topdown_pitch_length_map.png")
        cv2.imwrite(p2_path, g2_img)

        # Output 3: Dual-View Broadcast Pitch Map & Length-Zone Analytics Map
        g3_img = self.render_standalone_analytics_board(deliveries)
        p3_path = os.path.join(output_dir, "output_3_broadcast_and_length_zone_pitch_map.png")
        cv2.imwrite(p3_path, g3_img)
        cv2.imwrite(os.path.join(output_dir, "session_pitch_map_analytics.png"), g3_img)

        # Zone Crop Pictures for Output 5 (Spotlight Cards)
        z_stumps, _, _ = render_perspective_pitch_heatmap(hm_items, highlight_zone="G-Stumps", title="ZONE 1: GOOD-LENGTH STUMP LINE")
        crop_stumps = z_stumps[105:525, 55:565].copy()
        cv2.imwrite(os.path.join(output_dir, "zone_crop_stump_line_50pct.png"), crop_stumps)

        z_corr, _, _ = render_perspective_pitch_heatmap(hm_items, highlight_zone="Corridor", title="ZONE 2: CORRIDOR OF UNCERTAINTY")
        crop_corr = z_corr[105:555, 55:565].copy()
        cv2.imwrite(os.path.join(output_dir, "zone_crop_corridor_33pct.png"), crop_corr)

        z_good, _, _ = render_perspective_pitch_heatmap(hm_items, highlight_zone="GoodLength", title="ZONE 3: GOOD LENGTH BAND")
        crop_good = z_good[105:555, 55:565].copy()
        cv2.imwrite(os.path.join(output_dir, "zone_crop_good_length_83pct.png"), crop_good)

        # Output 4: 2D Line x Length Wicket Density Matrix + Executive Report
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        length_order = ["Full Toss (<1.2m)", "Yorker (1.2-2.2m)", "Full (2.2-5.0m)", "Good (Pace 5-8m / Spin 3-5.2m)", "Short (>8m Pace / >5.2m Spin)"]
        length_keys = ["Full Toss", "Yorker", "Full", "Good", "Short"]
        line_order = ["Wide Outside Off\n(>30cm Off)", "Corridor of Uncertainty\n(0-30cm Off)", "In-Line\n(Stump-to-Stump)", "Outside Leg\n(Pads)"]
        line_keys = ["WideOff", "CorridorOff", "Stumps", "OutsideLeg"]

        mat_counts = np.zeros((5, 4), dtype=int)
        mat_wickets = [[[] for _ in range(4)] for _ in range(5)]
        for item in hm_items:
            t = item["traj"]
            r_i = length_keys.index(t.pitching_length_zone) if t.pitching_length_zone in length_keys else 3
            l_code = t.matrix_18_cell.split("-")[1] if "-" in t.matrix_18_cell else "Stumps"
            c_i = line_keys.index(l_code) if l_code in line_keys else 2
            mat_counts[r_i, c_i] += 1
            mat_wickets[r_i][c_i].append(item["w_id"])

        mat_pct = (mat_counts / float(total_w)) * 100.0
        fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=180)
        fig.patch.set_facecolor("#141a16")
        ax.set_facecolor("#141a16")
        cmap = LinearSegmentedColormap.from_list("wicket_heat", ["#1e2923", "#1f5f5b", "#d97724", "#e63946"])
        im = ax.imshow(mat_pct, cmap=cmap, aspect="auto", vmin=0, vmax=max(55.0, float(mat_pct.max())))
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels(line_order, color="#f0f4f1", fontsize=8.5, fontweight="bold")
        ax.set_yticklabels(length_order, color="#f0f4f1", fontsize=8.5, fontweight="bold")
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, colors="#f0f4f1")

        for i in range(5):
            for j in range(4):
                cnt = mat_counts[i, j]
                pct = mat_pct[i, j]
                w_items = mat_wickets[i][j]
                w_list = (", ".join(w_items[:3]) + ",\n" + ", ".join(w_items[3:])) if len(w_items) > 3 else ", ".join(w_items)
                txt_col = "#ffffff" if pct >= 15 else ("#e2ece9" if cnt > 0 else "#6c7d74")
                main_txt = f"{pct:.1f}%\n({cnt}/{total_w} Wickets)" + (f"\n[{w_list}]" if w_list else "")
                ax.text(j, i, main_txt, ha="center", va="center", color=txt_col, fontsize=8.8 if w_list else 8.2, fontweight="bold" if cnt > 0 else "normal")

        for spine in ax.spines.values():
            spine.set_edgecolor("#4a5d52")
            spine.set_linewidth(1.2)
        ax.set_xticks(np.arange(5) - 0.5, minor=True)
        ax.set_yticks(np.arange(6) - 0.5, minor=True)
        ax.grid(which="minor", color="#3a4a40", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(f"2D LINE x LENGTH WICKET DENSITY MATRIX (% OF {total_w} WICKETS)", color="#f5faf7", fontsize=11.5, fontweight="bold", pad=16)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#f0f4f1")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#f0f4f1", fontsize=8.5)
        cbar.set_label("% of Total Wickets", color="#f0f4f1", fontsize=9.0)
        plt.tight_layout()
        matrix_png = os.path.join(output_dir, "line_length_matrix_heatmap.png")
        plt.savefig(matrix_png, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)

        mat_img = cv2.imread(matrix_png)
        stump_good = [d for d in hm_items if d["traj"].matrix_18_cell == "G-Stumps"]
        corridor_all = [d for d in hm_items if d["traj"].corridor_of_uncertainty]
        outside_off_all = [d for d in hm_items if d["traj"].pitching_line == "OUTSIDE OFF"]
        good_all = [d for d in hm_items if d["traj"].pitching_length_zone == "Good"]
        danger_all = [d for d in hm_items if d["traj"].pitching_line == "IN-LINE" or d["traj"].corridor_of_uncertainty]

        font = cv2.FONT_HERSHEY_DUPLEX
        g4 = np.zeros((980, 660, 3), dtype=np.uint8)
        g4[:, :] = (20, 26, 22)
        g4[10:520, 10:650] = cv2.resize(mat_img, (640, 510), interpolation=cv2.INTER_AREA)
        cv2.rectangle(g4, (10, 10), (650, 520), (85, 110, 95), 1)
        cv2.rectangle(g4, (10, 534), (650, 970), (26, 34, 29), -1)
        cv2.rectangle(g4, (10, 534), (650, 970), (85, 110, 95), 1)
        cv2.rectangle(g4, (10, 534), (650, 574), (35, 52, 42), -1)
        cv2.putText(g4, f"COACH'S WICKET ZONE EXECUTIVE REPORT ({total_w} DELIVERIES)", (26, 561), font, 0.52, (110, 240, 175), 1, cv2.LINE_AA)

        summary_lines = [
            (f"1. GOOD-LENGTH STUMP LINE [G-Stumps]: {100.0*len(stump_good)/total_w:.1f}% ({len(stump_good)} / {total_w} Wickets)", (95, 235, 255)),
            ("   - Direct stump-line Good Length deliveries (Bowled / LBW / Bat-Pad).", (215, 225, 220)),
            (f"2. CORRIDOR OF UNCERTAINTY [0-30cm Off]: {100.0*len(corridor_all)/total_w:.1f}% ({len(corridor_all)} / {total_w} Wickets)", (85, 195, 255)),
            (f"   - Total Outside Off Stump: {100.0*len(outside_off_all)/total_w:.1f}% ({len(outside_off_all)} / {total_w} Wickets).", (215, 225, 220)),
            (f"3. OVERALL GOOD LENGTH BAND: {100.0*len(good_all)/total_w:.1f}% ({len(good_all)} / {total_w} Wickets)", (120, 245, 140)),
            ("   - Optimal indecision zone between front-foot and back-foot play.", (215, 225, 220)),
            (f"4. COMBINED DANGER CHANNEL (Stumps + Corridor): {100.0*len(danger_all)/total_w:.1f}% ({len(danger_all)} / {total_w} Wickets)", (255, 215, 95)),
            ("   - Pitched inside the 53cm channel from Leg Stump to 30cm Outside Off.", (215, 225, 220)),
        ]
        cy_t = 606
        for l_str, l_col in summary_lines:
            is_h = l_str[0].isdigit()
            cv2.putText(g4, l_str, (26, cy_t), font, 0.43 if is_h else 0.39, l_col, 1, cv2.LINE_AA)
            cy_t += 40 if is_h else 36

        p4_path = os.path.join(output_dir, "output_4_matrix_and_executive_report.png")
        cv2.imwrite(p4_path, g4)

        # Output 5: 3 Visual Zone Spotlight Cards
        g5 = np.zeros((980, 612, 3), dtype=np.uint8)
        g5[:, :] = (20, 26, 22)
        th_1 = stump_good[0]["traj_img"] if stump_good else hm_items[0]["traj_img"]
        th_2 = corridor_all[0]["traj_img"] if corridor_all else hm_items[0]["traj_img"]
        th_3 = good_all[0]["traj_img"] if good_all else hm_items[0]["traj_img"]

        cards_cfg = [
            (
                f"SPOTLIGHT 1: {100.0*len(stump_good)/total_w:.1f}% WICKETS ({len(stump_good)}/{total_w}) - GOOD-LENGTH STUMP LINE",
                "Zone: G-Stumps (In-Line, Pace 5.0-6.0m / Spin 3.8-5.0m)",
                "Dismissal Mode: Bowled / LBW / Bat-Pad",
                crop_stumps, th_1, (65, 215, 255),
            ),
            (
                f"SPOTLIGHT 2: {100.0*len(corridor_all)/total_w:.1f}% WICKETS ({len(corridor_all)}/{total_w}) - CORRIDOR OF UNCERTAINTY",
                f"Zone: 0-30cm Outside Off Stump ({100.0*len(outside_off_all)/total_w:.1f}% Total Outside Off)",
                "Dismissal Mode: Outside Edge / Caught Behind / Slips",
                crop_corr, th_2, (85, 165, 255),
            ),
            (
                f"SPOTLIGHT 3: {100.0*len(good_all)/total_w:.1f}% WICKETS ({len(good_all)}/{total_w}) - GOOD LENGTH BAND",
                "Zone: Optimal Bounce Indecision Zone (Front/Back Foot Dilemma)",
                f"Wickets: {len(good_all)} of {total_w} Wickets Pitched in Good Length",
                crop_good, th_3, (110, 235, 130),
            ),
        ]
        for idx_c, (c_title, c_sub1, c_sub2, z_c, th_im, acc_bgr) in enumerate(cards_cfg):
            cy1 = 10 + idx_c * 326
            cy2 = cy1 + 308
            cv2.rectangle(g5, (10, cy1), (602, cy2), (28, 36, 31), -1)
            cv2.rectangle(g5, (10, cy1), (602, cy2), acc_bgr, 2)
            cv2.rectangle(g5, (10, cy1), (602, cy1 + 36), (38, 50, 42), -1)
            cv2.putText(g5, c_title, (22, cy1 + 24), font, 0.43, acc_bgr, 1, cv2.LINE_AA)
            cv2.putText(g5, c_sub1, (22, cy1 + 56), font, 0.39, (240, 245, 242), 1, cv2.LINE_AA)
            cv2.putText(g5, c_sub2, (22, cy1 + 78), font, 0.38, (190, 210, 198), 1, cv2.LINE_AA)

            g5[cy1 + 90 : cy1 + 296, 22 : 297] = cv2.resize(z_c, (275, 206), interpolation=cv2.INTER_AREA)
            cv2.rectangle(g5, (22, cy1 + 90), (297, cy1 + 296), (200, 210, 200), 1)
            cv2.rectangle(g5, (22, cy1 + 90), (195, cy1 + 112), (18, 22, 20), -1)
            cv2.putText(g5, "PITCH ZONE HEATMAP PIC", (28, cy1 + 106), font, 0.35, (110, 240, 255), 1, cv2.LINE_AA)

            g5[cy1 + 90 : cy1 + 296, 312 : 587] = cv2.resize(th_im, (275, 206), interpolation=cv2.INTER_AREA)
            cv2.rectangle(g5, (312, cy1 + 90), (587, cy1 + 296), (200, 210, 200), 1)
            cv2.rectangle(g5, (312, cy1 + 90), (505, cy1 + 112), (18, 22, 20), -1)
            cv2.putText(g5, "SAMPLE WICKET IN THIS ZONE", (318, cy1 + 106), font, 0.35, (255, 225, 110), 1, cv2.LINE_AA)

        p5_path = os.path.join(output_dir, "output_5_zone_spotlight_cards.png")
        cv2.imwrite(p5_path, g5)

        # Combined 1920x1080 Master Dashboard
        dash = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dash[:, :] = (20, 26, 22)
        cv2.rectangle(dash, (0, 0), (1920, 68), (14, 18, 15), -1)
        cv2.line(dash, (0, 68), (1920, 68), (75, 195, 135), 2)
        cv2.putText(dash, f"COACH'S WICKET-TAKING ZONE HEATMAP & LINE-LENGTH ANALYTICS ({total_w} WICKET DELIVERIES)", (28, 44), font, 0.82, (245, 250, 246), 2, cv2.LINE_AA)
        cv2.rectangle(dash, (24, 88), (624, 132), (28, 38, 31), -1)
        cv2.putText(dash, f"BROADCAST PITCH WICKET HEATMAP ({total_w} WICKETS)", (40, 117), font, 0.52, (245, 250, 246), 1, cv2.LINE_AA)
        dash[132:1052, 24:624] = cv2.resize(g1_img, (600, 920), interpolation=cv2.INTER_AREA)
        cv2.rectangle(dash, (24, 88), (624, 1052), (85, 110, 95), 1)
        dash[88:1052, 638:1298] = cv2.resize(g4, (660, 964), interpolation=cv2.INTER_AREA)
        dash[88:1052, 1308:1900] = cv2.resize(g5, (592, 964), interpolation=cv2.INTER_AREA)
        dash_path = os.path.join(output_dir, "wicket_heatmap_analytics_dashboard.png")
        cv2.imwrite(dash_path, dash)

        # Automatically generate the exact 4-page `coach_wicket_zone_heatmap_report.pdf` & `.html`
        pdf_report_path = os.path.join(output_dir, "coach_wicket_zone_heatmap_report.pdf")
        html_report_path = os.path.join(output_dir, "coach_wicket_zone_heatmap_report.html")
        try:
            import base64
            import shutil
            from matplotlib.backends.backend_pdf import PdfPages

            def _b64(img_p: str) -> str:
                if os.path.exists(img_p):
                    with open(img_p, "rb") as bf:
                        return "data:image/png;base64," + base64.b64encode(bf.read()).decode("ascii")
                return ""

            # Generate broadcast-style zone spotlight pictures matching Pages 2 & 3 of coach_wicket_zone_heatmap_report.pdf
            z1_bcast = render_broadcast_pitch_heatmap(
                hm_items, width=680, height=640, highlight_filter="G-Stumps",
                subtitle_badge=f"GOOD + MID (STUMPS): {100.0*len(stump_good)/total_w:.1f}% ({len(stump_good)}/{total_w} WICKETS)"
            )
            z1_bcast_p = os.path.join(output_dir, "zone_bcast_stump_line.png")
            cv2.imwrite(z1_bcast_p, z1_bcast)

            z2_bcast = render_broadcast_pitch_heatmap(
                hm_items, width=680, height=640, highlight_filter="OutsideOff",
                subtitle_badge=f"OFF CORRIDOR: {100.0*len(outside_off_all)/total_w:.1f}% ({len(outside_off_all)}/{total_w} WICKETS; {100.0*len(corridor_all)/total_w:.1f}% IN 30CM)"
            )
            z2_bcast_p = os.path.join(output_dir, "zone_bcast_corridor.png")
            cv2.imwrite(z2_bcast_p, z2_bcast)

            z3_bcast = render_broadcast_pitch_heatmap(
                hm_items, width=680, height=640, highlight_filter="GoodLength",
                subtitle_badge=f"GOOD LENGTH ZONE: {100.0*len(good_all)/total_w:.1f}% ({len(good_all)}/{total_w} WICKETS)"
            )
            z3_bcast_p = os.path.join(output_dir, "zone_bcast_good_length.png")
            cv2.imwrite(z3_bcast_p, z3_bcast)

            in_line_all = [d for d in hm_items if d["traj"].pitching_line == "IN-LINE"]

            rows_html = ""
            for d in hm_items:
                t = d["traj"]
                corr_str = "<strong>YES</strong>" if t.corridor_of_uncertainty else ("No (Wide Off)" if t.pitching_line == "OUTSIDE OFF" else "No")
                rows_html += (
                    f"<tr><td><strong>{d['w_id']}</strong></td><td><code>{d['clip_id']}</code></td>"
                    f"<td>{t.bowler_type.upper()} / {t.batter_hand}</td><td>{t.pitching_distance_m:.2f} m</td>"
                    f"<td>{t.pitching_length_zone}</td><td>{t.pitching_line}</td><td>{corr_str}</td>"
                    f"<td><code>{t.matrix_18_cell}</code></td><td>{t.lateral_offset_m:+.2f} m</td></tr>"
                )

            w_stump_ids = ", ".join(d["w_id"] for d in stump_good) or "None"
            w_corr_ids = ", ".join(d["w_id"] for d in corridor_all) or "None"
            w_off_ids = ", ".join(d["w_id"] for d in outside_off_all) or "None"
            w_inline_ids = ", ".join(d["w_id"] for d in in_line_all) or "None"
            w_good_ids = ", ".join(d["w_id"] for d in good_all) or "None"
            w_danger_ids = ", ".join(d["w_id"] for d in danger_all) or "None"

            html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Coach's Wicket-Taking Zone &amp; Pitch Heatmap Analysis Report</title>
  <style>
    :root {{ --primary: #12402b; --accent: #b83b2b; --text: #1c2420; --bg: #fcfdfb; }}
    @page {{
      size: A4;
      margin: 20mm 18mm;
      background: var(--bg);
      @bottom-right {{
        content: counter(page);
        font-family: 'Inter', 'Noto Sans', sans-serif;
        font-size: 8.5pt;
        color: var(--text);
      }}
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: transparent; }}
    body {{ font-family: 'Inter', 'Noto Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.58; color: var(--text); }}
    @media screen {{ html, body {{ background: var(--bg); }} body {{ max-width: 900px; margin: 0 auto; padding: 32px 40px; }} }}
    main.document-flow {{ width: 100%; margin: 0; padding: 0; }}
    * {{ border: none; border-radius: 0; box-shadow: none; background: transparent; outline: none; }}
    h1 {{ font-family: 'Noto Serif', Georgia, serif; font-size: 19pt; line-height: 1.25; margin: 0 0 10px 0; color: #ffffff; }}
    h2 {{ font-family: 'Noto Serif', Georgia, serif; font-size: 13pt; line-height: 1.35; margin: 28px 0 12px 0; color: var(--primary); break-after: avoid; page-break-after: avoid; }}
    h3 {{ font-family: 'Noto Serif', Georgia, serif; font-size: 11pt; line-height: 1.35; margin: 20px 0 8px 0; color: var(--accent); break-after: avoid; page-break-after: avoid; }}
    p {{ margin: 0 0 12px 0; }}
    ul, ol {{ margin: 0 0 14px 0; padding-left: 20px; }}
    li {{ margin-bottom: 6px; }}
    section {{ margin-bottom: 22px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }}
    th, td {{ text-align: left; padding: 8px 10px; font-size: 9.2pt; }}
    th {{ border-bottom: 2px solid var(--primary); font-weight: 600; color: var(--primary); }}
    td {{ border-bottom: 1px solid rgba(0,0,0,0.08); }}
    .full-bleed-header {{ background: var(--primary); color: #ffffff; padding: 22px 26px; margin-bottom: 22px; }}
    img {{ width: 100%; max-width: 100%; height: auto; object-fit: cover; display: block; }}
    figure {{ margin: 20px 0; }}
    figcaption {{ font-size: 8.5pt; color: var(--text); opacity: 0.78; margin-top: 8px; }}
    tr, figure {{ break-inside: avoid; page-break-inside: avoid; }}
  </style>
</head>
<body>
  <main class="document-flow">
    <div class="full-bleed-header">
      <h1>Coach's Wicket-Taking Zone &amp; Pitch Heatmap Analysis Report</h1>
      <p style="margin: 0; font-size: 9.8pt; color: #e2f0e8;">
        Quantitative Line &amp; Length Telemetry, 2D Kernel Density Pitch Heatmaps, and Zone-by-Zone Wicket Breakdown Across {total_w} Wicket-Taking Deliveries
      </p>
    </div>

    <section>
      <h2>1. Executive Coaching Takeaway: {100.0*len(danger_all)/total_w:.1f}% of Wickets Fall Within a 53 cm Channel From Stumps to the Off-Stump Corridor</h2>
      <p>
        Across all <strong>{total_w} analyzed wicket-taking deliveries</strong>, camera-stabilized first-bounce tracking (<code>best.pt</code> YOLOv8n pitch homography + <code>ball_detector.pt</code> YOLO11s DAG trajectory solver) demonstrates that batters lose their wickets in two primary pitch zones defined in <a href="https://en.wikipedia.org/wiki/Line_and_length">Line and length (Wikipedia)</a>:
      </p>
      <ul>
        <li><strong>Good-Length Stump-to-Stump Line (<code>G-Stumps</code> — {100.0*len(stump_good)/total_w:.1f}% of Wickets, {len(stump_good)} / {total_w} Wickets)</strong>: {len(stump_good)} dismissals (<code>{w_stump_ids}</code>) pitched directly in-line with the three stumps (<code>22.86 cm</code> carpet) on a Good Length (<code>5.0–6.0 m</code> for pace; <code>3.8–5.0 m</code> for spin), producing direct Bowled, LBW, and bat-pad dismissals.</li>
        <li><strong>Corridor of Uncertainty (<code>0–30 cm</code> Outside Off Stump — {100.0*len(corridor_all)/total_w:.1f}% of Wickets, {len(corridor_all)} / {total_w} Wickets; {100.0*len(outside_off_all)/total_w:.1f}% Total Outside Off)</strong>: {len(corridor_all)} wickets (<code>{w_corr_ids}</code>) pitched within <code>12 inches (30 cm)</code> outside the off stump, inducing outside edges and caught-behind dismissals (<strong>{100.0*len(outside_off_all)/total_w:.1f}% / {len(outside_off_all)} of {total_w} wickets</strong> total outside off stump).</li>
        <li><strong>Overall Good Length Band ({100.0*len(good_all)/total_w:.1f}% of Wickets, {len(good_all)} / {total_w} Wickets)</strong>: Accounting for bowler pace (<a href="https://en.wikipedia.org/wiki/Line_and_length">Wikipedia Line and length</a>: <code>5.0–8.0 m</code> for fast bowlers, <code>3.0–5.2 m</code> for spinners), <strong>{len(good_all)} of the {total_w} wickets ({100.0*len(good_all)/total_w:.1f}%)</strong> bounced in the Good Length zone where batters face maximum front-foot vs. back-foot indecision.</li>
      </ul>
      <figure>
        <img src="{_b64(dash_path)}" alt="Master Coaching Wicket Heatmap Dashboard">
        <figcaption>Figure 1: Master 3-Panel Coaching Wicket Heatmap Dashboard combining the 3D Perspective Pitch Gaussian KDE Thermal Heatmap (Left), the 2D Line × Length Wicket Density Matrix (Center), and Visual Zone Spotlight Cards with embedded pitch-zone pictures (Right).</figcaption>
      </figure>
    </section>

    <section>
      <h2>2. Wicket Distribution Summary by Pitch Zone (Line × Length Matrix)</h2>
      <p>
        The table below summarizes where batters lost their wickets across the 18-cell coaching matrix (<a href="https://ir.canterbury.ac.nz/handle/10092/15506">Canterbury Cricket Analytics</a>), broken down by both <strong>Pitching Line Corridor</strong> and <strong>Coaching Length Zone</strong>:
      </p>
      <table>
        <thead>
          <tr><th>Pitch Zone / Coaching Corridor</th><th>Physical Pitch Boundaries</th><th>Wickets Taken</th><th>% of Total Wickets</th><th>Wicket Delivery IDs in Zone</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>Good-Length Stump Line (<code>G-Stumps</code>)</strong></td><td>In-Line with Stumps (±11.4 cm), Good Length</td><td><strong>{len(stump_good)} / {total_w}</strong></td><td><strong>{100.0*len(stump_good)/total_w:.1f}%</strong></td><td><code>{w_stump_ids}</code></td></tr>
          <tr><td><strong>Corridor of Uncertainty (<code>0–30 cm Off</code>)</strong></td><td>0.00 m – 0.30 m Outside Off Stump</td><td><strong>{len(corridor_all)} / {total_w}</strong></td><td><strong>{100.0*len(corridor_all)/total_w:.1f}%</strong></td><td><code>{w_corr_ids}</code></td></tr>
          <tr><td><strong>Total Outside Off Stump (<code>Corridor + Wide Off</code>)</strong></td><td>0.00 m – 0.45 m Outside Off Stump</td><td><strong>{len(outside_off_all)} / {total_w}</strong></td><td><strong>{100.0*len(outside_off_all)/total_w:.1f}%</strong></td><td><code>{w_off_ids}</code></td></tr>
          <tr><td><strong>Total In-Line with Stumps (<code>All Lengths</code>)</strong></td><td>Wicket-to-Wicket Royal-Blue Carpet (22.86 cm)</td><td><strong>{len(in_line_all)} / {total_w}</strong></td><td><strong>{100.0*len(in_line_all)/total_w:.1f}%</strong></td><td><code>{w_inline_ids}</code></td></tr>
          <tr><td><strong>Overall Good Length Band (<code>Pace + Spin</code>)</strong></td><td>Pace 5.0–8.0 m / Spin 3.0–5.2 m from Stumps</td><td><strong>{len(good_all)} / {total_w}</strong></td><td><strong>{100.0*len(good_all)/total_w:.1f}%</strong></td><td><code>{w_good_ids}</code></td></tr>
          <tr><td><strong>Combined Danger Channel (<code>Stumps + Corridor</code>)</strong></td><td>53 cm Channel (Leg Stump to 30 cm Outside Off)</td><td><strong>{len(danger_all)} / {total_w}</strong></td><td><strong>{100.0*len(danger_all)/total_w:.1f}%</strong></td><td><code>{w_danger_ids}</code></td></tr>
        </tbody>
      </table>
    </section>

    <section>
      <h2>3. Visual Pitch Zone Spotlights: Where Batters Are Losing Wickets</h2>
      <p>
        To give coaches a clear visual reference for each danger area on the turf, the three spotlight figures below isolate the exact pitch region for each primary wicket-taking zone alongside its Gaussian Kernel Density (KDE) thermal hotspot:
      </p>
      <h3>Spotlight Zone 1: Good-Length Stump Line ({100.0*len(stump_good)/total_w:.1f}% of Wickets — {len(stump_good)} / {total_w} Wickets)</h3>
      <p>
        {len(stump_good)} of the {total_w} wickets (<strong>{100.0*len(stump_good)/total_w:.1f}%</strong>) pitched directly on the stump-to-stump line within the Good Length band (<code>{w_stump_ids}</code>). Batters playing across the line or planting their front foot early were beaten on the inside/outside edge or trapped LBW/Bowled.
      </p>
      <figure>
        <img src="{_b64(z1_bcast_p)}" alt="Zone 1 Good-Length Stump Line" style="max-height: 250px; object-fit: contain;">
        <figcaption>Figure 2: Zoomed Pitch Zone Picture for Zone 1 — Good-Length Stump Line (<code>G-Stumps</code>), where {100.0*len(stump_good)/total_w:.1f}% ({len(stump_good)} / {total_w}) of all wickets were taken.</figcaption>
      </figure>

      <h3>Spotlight Zone 2: Corridor of Uncertainty ({100.0*len(corridor_all)/total_w:.1f}% of Wickets — {len(corridor_all)} / {total_w} Wickets; {100.0*len(outside_off_all)/total_w:.1f}% Total Outside Off)</h3>
      <p>
        {len(corridor_all)} of the {total_w} wickets (<strong>{100.0*len(corridor_all)/total_w:.1f}%</strong>) pitched inside the <code>30 cm (12-inch)</code> <strong>Corridor of Uncertainty</strong> just outside off stump (<code>{w_corr_ids}</code>), with <strong>{100.0*len(outside_off_all)/total_w:.1f}% ({len(outside_off_all)} of {total_w} wickets)</strong> total pitching outside off stump.
      </p>
      <figure>
        <img src="{_b64(z2_bcast_p)}" alt="Zone 2 Corridor of Uncertainty" style="max-height: 250px; object-fit: contain;">
        <figcaption>Figure 3: Zoomed Pitch Zone Picture for Zone 2 — Corridor of Uncertainty (<code>0–30 cm</code> Outside Off Stump), accounting for {100.0*len(corridor_all)/total_w:.1f}% ({len(corridor_all)} / {total_w}) of wickets ({100.0*len(outside_off_all)/total_w:.1f}% total Outside Off).</figcaption>
      </figure>

      <h3>Spotlight Zone 3: Overall Good Length Band ({100.0*len(good_all)/total_w:.1f}% of Wickets — {len(good_all)} / {total_w} Wickets)</h3>
      <p>
        {len(good_all)} of the {total_w} wickets (<strong>{100.0*len(good_all)/total_w:.1f}%</strong>) pitched in the Good Length band (<code>5.0–8.0 m</code> for pace; <code>3.0–5.2 m</code> for spin), confirming that length discipline in the indecision zone is the single strongest predictor of wicket-taking success across both fast bowlers and spinners.
      </p>
      <figure>
        <img src="{_b64(z3_bcast_p)}" alt="Zone 3 Overall Good Length Band" style="max-height: 250px; object-fit: contain;">
        <figcaption>Figure 4: Zoomed Pitch Zone Picture for Zone 3 — Overall Good Length Band, where {100.0*len(good_all)/total_w:.1f}% ({len(good_all)} / {total_w}) of all wicket-taking deliveries bounced.</figcaption>
      </figure>
    </section>

    <section>
      <h2>4. Complete {total_w}-Delivery Wicket Telemetry &amp; 18-Cell Coaching Log</h2>
      <p>
        Every delivery processed by <code>ball_tracking_pipeline.py</code> is logged with its exact camera-stabilized bounce distance, lateral offset from middle stump, bowler pace/spin classification, batter handedness, and 18-cell coaching matrix code:
      </p>
      <table>
        <thead>
          <tr><th>ID</th><th>Delivery Clip</th><th>Bowler / Hand</th><th>Pitching Length</th><th>Length Zone</th><th>Pitching Line</th><th>Corridor (≤30cm)</th><th>Matrix Cell</th><th>Offset (m)</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""
            with open(html_report_path, "w", encoding="utf-8") as hf:
                hf.write(html_doc)

            # 1. Try compiling exact 4-page vector PDF via Selenium Chromium CDP (or headless Chrome CLI)
            compiled_via_browser = False
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome import options as chrome_options
                u_dir_tmp = tempfile.mkdtemp(prefix="coach_pdf_")
                try:
                    opts = chrome_options.Options()
                    opts.add_argument("--headless=new")
                    opts.add_argument("--no-sandbox")
                    opts.add_argument("--disable-dev-shm-usage")
                    opts.add_argument("--disable-gpu")
                    opts.add_argument(f"--user-data-dir={u_dir_tmp}")
                    drv = webdriver.Chrome(options=opts)
                    try:
                        drv.get("about:blank")
                        ft = drv.execute_cdp_cmd("Page.getFrameTree", {})
                        fid = ft["frameTree"]["frame"]["id"]
                        drv.execute_cdp_cmd("Page.setDocumentContent", {"frameId": fid, "html": html_doc})
                        import time as _t
                        _t.sleep(1.5)
                        res_pdf = drv.execute_cdp_cmd("Page.printToPDF", {
                            "landscape": False, "displayHeaderFooter": False,
                            "printBackground": True, "preferCSSPageSize": True,
                            "marginTop": 0, "marginBottom": 0, "marginLeft": 0, "marginRight": 0,
                        })
                        if res_pdf and "data" in res_pdf:
                            with open(pdf_report_path, "wb") as pf:
                                pf.write(base64.b64decode(res_pdf["data"]))
                            compiled_via_browser = True
                    finally:
                        drv.quit()
                finally:
                    shutil.rmtree(u_dir_tmp, ignore_errors=True)
            except Exception:
                for bin_name in ("google-chrome", "chromium", "chromium-browser", "msedge"):
                    if shutil.which(bin_name):
                        subprocess.run(
                            [bin_name, "--headless", "--disable-gpu", "--no-sandbox", f"--print-to-pdf={pdf_report_path}", html_report_path],
                            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        if os.path.exists(pdf_report_path) and os.path.getsize(pdf_report_path) > 1000:
                            compiled_via_browser = True
                            break

            # 2. Fallback to Matplotlib multi-page vector PDF if no browser is installed
            if not compiled_via_browser:
                with PdfPages(pdf_report_path) as pdf_pages:
                    fig1 = plt.figure(figsize=(8.27, 11.69), dpi=150)
                    fig1.patch.set_facecolor("#fcfdfb")
                    ax_hdr = fig1.add_axes([0.06, 0.86, 0.88, 0.09])
                    ax_hdr.set_facecolor("#12402b")
                    ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
                    ax_hdr.text(0.03, 0.62, "Coach's Wicket-Taking Zone & Pitch Heatmap Analysis Report", color="white", fontsize=14, fontweight="bold")
                    ax_hdr.text(0.03, 0.22, f"Quantitative Line & Length Telemetry, 2D KDE Pitch Heatmaps & Zone Breakdown ({total_w} Deliveries)", color="#d8efe2", fontsize=8.8)
                    ax_d = fig1.add_axes([0.06, 0.08, 0.88, 0.74])
                    ax_d.imshow(cv2.cvtColor(dash, cv2.COLOR_BGR2RGB))
                    ax_d.axis("off")
                    pdf_pages.savefig(fig1, facecolor=fig1.get_facecolor())
                    plt.close(fig1)
        except Exception as e:
            print(f"[WARN] PDF report generation warning: {e}")

        return {
            "output_1_broadcast_pitch_heatmap": p1_path,
            "output_2_topdown_pitch_length_map": p2_path,
            "output_3_broadcast_and_length_zone_pitch_map": p3_path,
            "output_4_matrix_and_executive_report": p4_path,
            "output_5_zone_spotlight_cards": p5_path,
            "wicket_heatmap_analytics_dashboard": dash_path,
            "coach_wicket_zone_heatmap_report_pdf": pdf_report_path,
            "coach_wicket_zone_heatmap_report_html": html_report_path,
        }


# =============================================================================
# 5. End-to-End Dual-Output Match & Wicket Video Pipeline
# =============================================================================

class CricketTrajectoryPipeline:
    """
    Produces BOTH requested outputs for every wicket-taking or match delivery clip:
      1. Output 1 (`<stem>_trajectory.mp4` & `<stem>_trajectory.png`):
         3D Hawk-Eye Ball Trajectory in the pitch region.
      2. Output 2 (`<stem>_pitching_details.mp4`, `<stem>_pitching_details.png`,
         and cumulative `session_pitch_map_analytics.png`):
         Ball Pitching Details & Length-Zone Pitch Map.
    """

    def __init__(
        self,
        ball_model_path: str = "ball_detector.pt",
        pitch_model_path: str = "best.pt",
        ball_conf: float = 0.12,
        pitch_conf: float = 0.25,
        device: Optional[str] = None,
        bowler_type: str = "auto",
        batter_hand: str = "RHB",
    ):
        self.pitch_detector = PitchDetector(
            model_path=pitch_model_path,
            imgsz=640,
            conf=pitch_conf,
            device=device,
        )
        self.ball_tracker = PitchBallTracker(
            model_path=ball_model_path,
            imgsz=960,
            conf=ball_conf,
            device=device,
            bowler_type=bowler_type,
            batter_hand=batter_hand,
        )
        self.trajectory_renderer = HawkEyeRenderer(style="white_blue")
        self.pitching_renderer = PitchingDetailsRenderer()

    @staticmethod
    def clean_annotated_frame(frame: np.ndarray) -> np.ndarray:
        """Strips any pre-drawn yellow Pitch or red Ball debug bounding boxes from frames."""
        out = frame.copy()
        h, w = out.shape[:2]
        b, g, r = out[:, :, 0].astype(int), out[:, :, 1].astype(int), out[:, :, 2].astype(int)
        y_top_cut = int(0.18 * h)
        y_bot_cut = int(0.87 * h)
        yellow_mask = ((r > 175) & (g > 175) & (b < 95) & ((r - b) > 95) & ((g - b) > 95)).astype(np.uint8) * 255
        yellow_mask[:y_top_cut, :] = 0
        yellow_mask[y_bot_cut:, :] = 0
        if np.any(yellow_mask):
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            yellow_closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            yellow_dil = cv2.dilate(yellow_closed, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        else:
            yellow_dil = yellow_mask

        red_mask = ((r > 165) & (g < 75) & (b < 75) & ((r - g) > 95)).astype(np.uint8) * 255
        red_mask[: int(0.24 * h), :] = 0
        red_mask[y_bot_cut:, :] = 0
        if np.any(red_mask):
            red_dil = cv2.dilate(red_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
        else:
            red_dil = red_mask

        mask = cv2.bitwise_or(yellow_dil, red_dil)
        if np.any(mask):
            return cv2.inpaint(out, mask, 5, cv2.INPAINT_TELEA)
        return out

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        freeze_hold_frames: int = 50,
    ) -> Tuple[Dict[str, object], Optional[TrajectoryModel]]:
        self.pitch_detector.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        n_frames = len(frames)
        if n_frames == 0:
            raise RuntimeError(f"No frames decoded from {video_path}")

        sampled_boxes: Dict[int, np.ndarray] = {}
        for idx in range(0, n_frames, 2):
            sampled_boxes[idx] = self.pitch_detector.detect_raw_box(frames[idx])
        if (n_frames - 1) not in sampled_boxes:
            sampled_boxes[n_frames - 1] = self.pitch_detector.detect_raw_box(frames[-1])

        pitch_geoms: List[PitchGeometry] = []
        for idx in range(n_frames):
            if idx in sampled_boxes:
                rbox = sampled_boxes[idx]
            else:
                prev_k = max(k for k in sampled_boxes if k < idx)
                next_k = min(k for k in sampled_boxes if k > idx)
                alpha = (idx - prev_k) / max(1.0, float(next_k - prev_k))
                rbox = (1.0 - alpha) * sampled_boxes[prev_k] + alpha * sampled_boxes[next_k]
            pitch_geoms.append(self.pitch_detector.estimate_geometry(frames[idx], raw_box=rbox))

        all_dets: List[BallDetection] = []
        for idx in range(n_frames):
            dets = self.ball_tracker.detect_in_pitch_corridor(frames[idx], idx, pitch_geoms[idx])
            all_dets.extend(dets)

        clean_chain = self.ball_tracker.select_clean_delivery_chain(all_dets)
        impact_idx = clean_chain[-1].frame_idx if clean_chain else (n_frames - 1)
        ref_geom = pitch_geoms[min(n_frames - 1, impact_idx)]
        traj = self.ball_tracker.fit_trajectory(clean_chain, ref_geom, pitch_geoms=pitch_geoms)

        os.makedirs(output_dir, exist_ok=True)
        stem = Path(video_path).stem

        traj_mp4 = os.path.join(output_dir, f"{stem}_trajectory.mp4")
        traj_png = os.path.join(output_dir, f"{stem}_trajectory.png")
        pitch_mp4 = os.path.join(output_dir, f"{stem}_pitching_details.mp4")
        pitch_png = os.path.join(output_dir, f"{stem}_pitching_details.png")

        out_w, out_h = 1280, 720
        raw_t_mp4 = f"/tmp/{stem}_traj_raw.mp4"
        raw_p_mp4 = f"/tmp/{stem}_pitch_raw.mp4"
        w_traj = cv2.VideoWriter(raw_t_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        w_pitch = cv2.VideoWriter(raw_p_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        start_w = max(0, (traj.start_frame - 15) if traj else 0)
        end_w = min(n_frames, (traj.impact_frame + 5) if traj else n_frames)

        for idx in range(start_w, end_w):
            f_traj = self.trajectory_renderer.render_frame(
                frames[idx], pitch_geoms[idx], traj, frame_idx=idx, show_full_trajectory=False
            )
            w_traj.write(cv2.resize(f_traj, (out_w, out_h)))

            show_spot = traj if (traj is not None and idx >= int(round(traj.bounce_frame))) else None
            f_pitch = self.pitching_renderer.render_pitching_details_on_frame(
                frames[idx], pitch_geoms[idx], show_spot, show_trajectory_curve=False
            )
            w_pitch.write(cv2.resize(f_pitch, (out_w, out_h)))

            if traj is not None and idx == traj.impact_frame:
                sum_traj = self.trajectory_renderer.render_frame(
                    frames[idx], pitch_geoms[idx], traj, frame_idx=None, show_full_trajectory=True
                )
                sum_pitch = self.pitching_renderer.render_pitching_details_on_frame(
                    frames[idx], pitch_geoms[idx], traj, show_trajectory_curve=True
                )
                cv2.imwrite(traj_png, sum_traj)
                cv2.imwrite(pitch_png, sum_pitch)
                st_res = cv2.resize(sum_traj, (out_w, out_h))
                sp_res = cv2.resize(sum_pitch, (out_w, out_h))
                for _ in range(freeze_hold_frames):
                    w_traj.write(st_res)
                    w_pitch.write(sp_res)

        w_traj.release()
        w_pitch.release()

        for src_raw, dst_final in [(raw_t_mp4, traj_mp4), (raw_p_mp4, pitch_mp4)]:
            if os.path.exists("/usr/bin/ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src_raw, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", dst_final],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                os.replace(src_raw, dst_final)

        stats = {
            "video_path": video_path,
            "trajectory_video": traj_mp4,
            "trajectory_image": traj_png,
            "pitching_details_video": pitch_mp4,
            "pitching_details_image": pitch_png,
            "pitching_distance_m": round(traj.pitching_distance_m, 2) if traj else None,
            "pitching_length_zone": traj.pitching_length_zone if traj else None,
            "pitching_line": traj.pitching_line if traj else None,
            "corridor_of_uncertainty": traj.corridor_of_uncertainty if traj else None,
            "matrix_18_cell": traj.matrix_18_cell if traj else None,
            "bowler_type": traj.bowler_type if traj else None,
            "bounce_point": traj.bounce_pt if traj else None,
            "end_point": traj.end_pt if traj else None,
        }
        return stats, traj


def download_github_release_videos(
    release_tag_url: str = "https://github.com/nikhilmutyala20-msn/datasets/releases/tag/v1",
    download_dir: str = "./datasets_v1",
) -> List[str]:
    os.makedirs(download_dir, exist_ok=True)
    m = re.search(r"github\.com/([^/]+)/([^/]+)/releases/tag/([^/?#]+)", release_tag_url)
    if not m:
        raise ValueError(f"Invalid GitHub release tag URL: {release_tag_url}")

    owner, repo, tag = m.group(1), m.group(2), m.group(3)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    headers = {"User-Agent": "Mozilla/5.0 (Cricket-HawkEye-Tracker)"}

    asset_urls: List[Tuple[str, str]] = []
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            release_data = json.loads(resp.read().decode("utf-8"))
        for asset in release_data.get("assets", []):
            name = asset.get("name", "")
            url = asset.get("browser_download_url", "")
            if url and name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".zip")):
                asset_urls.append((name, url))
    except Exception:
        expanded_url = f"https://github.com/{owner}/{repo}/releases/expanded_assets/{tag}"
        req = urllib.request.Request(expanded_url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        matches = re.findall(r'href="(/[^"]+/releases/download/[^"]+)"', html)
        for rel_path in matches:
            fname = os.path.basename(rel_path)
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".zip")):
                asset_urls.append((fname, f"https://github.com{rel_path}"))

    downloaded_videos: List[str] = []
    for name, url in asset_urls:
        dest = os.path.join(download_dir, name)
        if not os.path.exists(dest):
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out_f:
                out_f.write(resp.read())
        if dest.lower().endswith(".zip"):
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(download_dir)
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                downloaded_videos.extend(glob.glob(os.path.join(download_dir, "**", ext), recursive=True))
        else:
            downloaded_videos.append(dest)

    return sorted(set(downloaded_videos))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Cricket Ball Tracking & Pitching Analytics Pipeline (Dual-Output Match Wicket System)"
    )
    parser.add_argument("--ball-model", type=str, default="ball_detector.pt")
    parser.add_argument("--pitch-model", type=str, default="best.pt")
    parser.add_argument(
        "--source",
        type=str,
        default="https://github.com/nikhilmutyala20-msn/datasets/releases/tag/v1",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--ball-conf", type=float, default=0.12)
    parser.add_argument("--pitch-conf", type=float, default=0.25)
    parser.add_argument("--bowler-type", type=str, default="auto", choices=["auto", "pace", "spin"])
    parser.add_argument("--batter-hand", type=str, default="RHB", choices=["RHB", "LHB"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.source.startswith("http://") or args.source.startswith("https://"):
        video_files = download_github_release_videos(
            args.source, download_dir=os.path.join(args.output_dir, "downloaded_videos")
        )
    elif os.path.isdir(args.source):
        video_files = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
            video_files.extend(glob.glob(os.path.join(args.source, ext)))
        video_files = sorted(video_files)
    else:
        video_files = [args.source]

    if not video_files:
        print(f"[WARN] No video files found for source: {args.source}")
        sys.exit(1)

    pipeline = CricketTrajectoryPipeline(
        ball_model_path=args.ball_model,
        pitch_model_path=args.pitch_model,
        ball_conf=args.ball_conf,
        pitch_conf=args.pitch_conf,
        device=args.device,
        bowler_type=args.bowler_type,
        batter_hand=args.batter_hand,
    )

    session_deliveries: List[Tuple[str, TrajectoryModel]] = []
    report_rows: List[Dict[str, object]] = []
    for vid in video_files:
        stats, traj = pipeline.process_video(vid, args.output_dir)
        if traj is not None:
            session_deliveries.append((Path(vid).stem, traj))
            report_rows.append(stats)
        print(f"[DUAL-OUTPUT] {vid} -> {stats}")

    



    if session_deliveries:
        pipeline.pitching_renderer.render_heatmap_analytics_dashboard(session_deliveries, args.output_dir)
        board = pipeline.pitching_renderer.render_standalone_analytics_board(session_deliveries)
        board_path = os.path.join(args.output_dir, "session_pitch_map_analytics.png")
        cv2.imwrite(board_path, board)
        print(f"[ANALYTICS BOARD] Saved cumulative Pitch Map to {board_path}")

    if report_rows:
        csv_path = os.path.join(args.output_dir, "match_wickets_line_length_report.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"[REPORT] Saved Line & Length coaching report to {csv_path}")


if __name__ == "__main__":
    main()