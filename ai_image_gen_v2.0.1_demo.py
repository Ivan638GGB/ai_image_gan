# ==============================================================================
# AI Image Generator (GAN)
# Copyright (C) 2026 Ivan Nidostup (GGB_638), Kryvyi Rih, Ukraine.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

import os
import io
import math
import threading
import random
import time
import sys
import copy
import re
import hashlib
import json
import logging
import getpass
import platform
import struct
import wave
from contextlib import nullcontext
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageTk, ImageFilter

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils


if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(APP_DIR, "dataset")
DATASET_VIDEO_DIR = os.path.join(APP_DIR, "dataset_video")
CHECKPOINT_DIR = os.path.join(APP_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(APP_DIR, "output")
MODELS_DIR = os.path.join(APP_DIR, "models")
VGG_MODELS_DIR = os.path.join(APP_DIR, "models_vgg")
LOG_FILE = os.path.join(APP_DIR, "training_log.txt")
PROFILES_FILE = os.path.join(APP_DIR, "profiles.json")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
HISTORY_FILE = os.path.join(APP_DIR, "training_history.json")
LEGAL_ACCEPTANCE_LOG = os.path.join(APP_DIR, "legal_acceptance_log.jsonl")

LATENT_DIM = 100
ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
ALLOWED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".wmv")
ALLOWED_MODEL_EXTENSIONS = (".pth", ".pt", ".safetensors", ".ckpt", ".bin")

# ── Дефолтные настройки ──────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    # Обучение
    "enable_grad_accum": True,
    "grad_accum_steps": 4,
    "enable_fp16": False,          # DirectML: пока нестабильно, по умолчанию выкл
    "enable_multi_gpu": False,
    "enable_fid_score": False,     # требует scipy + torchvision inception
    "fid_every_n_epochs": 10,
    "enable_early_stopping": True,
    "early_stopping_patience": 15,
    "lr_scheduler": "cosine",      # cosine / step / none
    "lr_warmup_epochs": 3,
    # Датасет
    "enable_phash_dedup": True,
    "phash_threshold": 8,
    "enable_clip_filter": False,   # требует openai-clip
    "clip_min_score": 0.2,
    "enable_face_crop": False,     # требует opencv
    "enable_balance_dataset": False,
    # Интерфейс
    "theme": "dark",
    "language": "ru",
    "enable_sound_notify": True,
    "enable_toast_notify": True,
    "enable_file_log": True,
    # Генерация
    "enable_onnx_export": True,
}


class AppSettings:
    """Глобальные настройки приложения — загружаются из settings.json."""
    def __init__(self):
        self._data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if os.path.isfile(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data.update(saved)
            except Exception:
                pass

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get(self, key, default=None):
        return self._data.get(key, DEFAULT_SETTINGS.get(key, default))

    def set(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)


SETTINGS = AppSettings()

# ── Файловый логгер ───────────────────────────────────────────────────────────
_file_logger = logging.getLogger("gan_trainer")
_file_logger.setLevel(logging.DEBUG)
if SETTINGS.get("enable_file_log"):
    try:
        _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        _file_logger.addHandler(_fh)
    except Exception:
        pass


def log_to_file(msg: str):
    if SETTINGS.get("enable_file_log"):
        _file_logger.info(msg)


# ── История обучений ──────────────────────────────────────────────────────────
def save_training_run(run_info: dict):
    history = []
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(run_info)
    history = history[-50:]  # храним последние 50 запусков
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_training_history() -> list:
    if os.path.isfile(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []
AD_URL_KEYWORDS = (
    "ad",
    "ads",
    "banner",
    "sponsor",
    "doubleclick",
    "pixel",
    "tracking",
    "logo",
    "avatar",
    "icon",
    "sprite",
)

LANG_MESSAGES = {
    "ru": {
        "status_ready": "Статус: готово",
        "status_low_data": "Статус: данных мало",
        "status_can_train": "Статус: можно обучать",
        "status_ready_train": "Статус: готово к обучению",
        "status_auto_start": "Статус: авто-режим запущен",
        "status_auto_stop": "Статус: авто-режим остановлен",
        "status_auto_done": "Статус: авто-режим завершен",
        "readiness_title": "Проверка готовности",
        "wait_title": "Подожди",
        "error_title": "Ошибка",
        "done_title": "Готово",
        "low_data": "Сначала скачайте хотя бы 20+ изображений.",
        "auto_done": "Готово! Данные скачаны, модель обучена и изображение сгенерировано.",
        "download_done": "Скачано: {count} изображений.",
        "train_done": "Обучение завершено.\nЧекпоинт: {path}",
        "gen_done": "Изображения сохранены:\n{path}",
        "running_task": "Сейчас уже выполняется задача.",
        "lang_switched": "Язык переключен: Русский",
        "status_running": "Статус: выполняется",
        "status_done": "Статус: готово",
    },
    "en": {
        "status_ready": "Status: ready",
        "status_low_data": "Status: low data",
        "status_can_train": "Status: can train",
        "status_ready_train": "Status: ready to train",
        "status_auto_start": "Status: auto mode started",
        "status_auto_stop": "Status: auto mode stopped",
        "status_auto_done": "Status: auto mode completed",
        "readiness_title": "Readiness check",
        "wait_title": "Wait",
        "error_title": "Error",
        "done_title": "Done",
        "low_data": "Download at least 20+ images first.",
        "auto_done": "Done! Data downloaded, model trained, image generated.",
        "download_done": "Downloaded: {count} images.",
        "train_done": "Training finished.\nCheckpoint: {path}",
        "gen_done": "Images saved:\n{path}",
        "running_task": "A task is already running.",
        "lang_switched": "Language switched: English",
        "status_running": "Status: running",
        "status_done": "Status: ready",
    },
}


def ensure_dirs() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(DATASET_VIDEO_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VGG_MODELS_DIR, exist_ok=True)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA)"

    try:
        import torch_directml  # type: ignore

        dml = torch_directml.device()
        return dml, "DirectML (AMD/Intel)"
    except Exception:
        return torch.device("cpu"), "CPU"


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Differentiable augmentations for better GAN stability on small datasets."""
    if x.ndim != 4:
        return x
    b, _, h, w = x.shape
    out = x

    # Color jitter in normalized [-1, 1] space.
    out = out + (torch.rand(b, 1, 1, 1, device=out.device) - 0.5) * 0.2
    mean_c = out.mean(dim=1, keepdim=True)
    out = (out - mean_c) * (torch.rand(b, 1, 1, 1, device=out.device) * 0.4 + 0.8) + mean_c
    mean_all = out.mean(dim=(1, 2, 3), keepdim=True)
    out = (out - mean_all) * (torch.rand(b, 1, 1, 1, device=out.device) * 0.4 + 0.8) + mean_all

    # Small random translation — без torch.roll (не поддерживается DirectML).
    # Используем padding + crop через F.pad что работает везде.
    max_shift = 2
    shifts_x = [random.randint(-max_shift, max_shift) for _ in range(b)]
    shifts_y = [random.randint(-max_shift, max_shift) for _ in range(b)]
    translated = []
    for i in range(b):
        img = out[i].unsqueeze(0)  # [1, C, H, W]
        sy, sx = shifts_y[i], shifts_x[i]
        # Pad и crop — эквивалент roll но без unsupported op
        pad_y = (max(0, sy), max(0, -sy))
        pad_x = (max(0, sx), max(0, -sx))
        img = F.pad(img, (pad_x[0], pad_x[1], pad_y[0], pad_y[1]), mode="replicate")
        # Crop обратно до исходного размера
        start_y = max(0, -sy)
        start_x = max(0, -sx)
        img = img[:, :, start_y:start_y+h, start_x:start_x+w]
        translated.append(img.squeeze(0))
    out = torch.stack(translated, dim=0)

    # Random cutout.
    cut = max(2, int(min(h, w) * 0.2))
    for i in range(b):
        cy = random.randint(0, max(0, h - 1))
        cx = random.randint(0, max(0, w - 1))
        y1 = max(0, cy - cut // 2)
        y2 = min(h, cy + cut // 2)
        x1 = max(0, cx - cut // 2)
        x2 = min(w, cx + cut // 2)
        out[i, :, y1:y2, x1:x2] = 0

    return torch.clamp(out, -1.0, 1.0)


def feature_matching_loss(netD, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Feature matching loss — дискриминатор как feature extractor.
    Генератор учится делать похожие на реальные промежуточные активации.
    Помогает с деталями без VGG, работает на DirectML."""
    real_feats = []
    fake_feats = []

    # Хукаем промежуточные слои дискриминатора
    hooks = []
    def make_hook(store):
        def hook(_, __, output):
            store.append(output.detach() if store is real_feats else output)
        return hook

    # Регистрируем хуки на Conv2d слои (не последний)
    conv_layers = [m for m in netD.modules() if isinstance(m, nn.Conv2d)]
    for layer in conv_layers[:-1]:  # пропускаем последний (classifier)
        hooks.append(layer.register_forward_hook(make_hook(real_feats)))

    with torch.no_grad():
        netD(real)
    for h in hooks: h.remove()
    hooks.clear()

    for layer in conv_layers[:-1]:
        hooks.append(layer.register_forward_hook(make_hook(fake_feats)))
    netD(fake)
    for h in hooks: h.remove()

    loss = torch.tensor(0.0, device=real.device)
    for rf, ff in zip(real_feats, fake_feats):
        if rf.shape == ff.shape:
            loss = loss + F.l1_loss(ff, rf.detach())
    return loss / max(len(real_feats), 1)


def diversity_regularizer(fake: torch.Tensor) -> torch.Tensor:
    """
    Penalize high similarity between samples in a batch.
    Lower value means more diverse generated images.
    """
    b = fake.size(0)
    if b < 2:
        return fake.new_tensor(0.0)
    flat = fake.view(b, -1)
    flat = F.normalize(flat, dim=1)
    sim = torch.matmul(flat, flat.t())
    # Use off-diagonal mean without boolean masking (more backend-friendly).
    diag_sum = torch.diagonal(sim, 0).sum()
    offdiag_sum = sim.sum() - diag_sum
    denom = float(b * (b - 1))
    return offdiag_sum / denom


class Generator(nn.Module):
    def __init__(self, image_size=64, latent_dim=LATENT_DIM, channels=3):
        super().__init__()
        assert image_size in (64, 128, 256), "Generator supports image_size 64, 128, or 256."
        self.image_size = image_size

        # GroupNorm вместо BatchNorm2d: стабильнее на DirectML/AMD/Intel,
        # не зависит от размера батча, лучше работает на малых датасетах.
        if image_size == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
                nn.GroupNorm(32, 512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        elif image_size == 128:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
                nn.GroupNorm(32, 1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 2048, 4, 1, 0, bias=False),
                nn.GroupNorm(32, 2048),
                nn.ReLU(True),
                nn.ConvTranspose2d(2048, 1024, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.GroupNorm(32, 64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=64, channels=3):
        super().__init__()
        assert image_size in (64, 128, 256), "Discriminator supports image_size 64, 128, or 256."
        self.image_size = image_size

        if image_size == 64:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
            )
        elif image_size == 128:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(1024, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
            )
        else:
            self.main = nn.Sequential(
                spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(1024, 2048, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(2048, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.main(x).view(-1)


class VideoGenerator(nn.Module):
    """Temporal GAN — DirectML совместимый.
    Вместо LSTM используем MLP который генерирует разные латентные векторы для каждого кадра.
    Это полностью работает на DirectML/AMD/Intel — нет 5D операций."""
    def __init__(self, latent_dim=100, image_size=64, n_frames=16):
        super().__init__()
        self.n_frames = n_frames
        self.image_size = image_size
        self.latent_dim = latent_dim

        # Для каждого кадра — отдельная линейная проекция из общего z
        # Это даёт "движение" — каждый кадр получает слегка другой вектор
        self.frame_mlp = nn.Sequential(
            nn.Linear(latent_dim + n_frames, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh(),
        )

        if image_size == 64:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False), nn.GroupNorm(32, 512), nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.GroupNorm(32, 256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.GroupNorm(32, 128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.GroupNorm(32, 64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh(),
            )
        else:  # 128
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False), nn.GroupNorm(32, 1024), nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), nn.GroupNorm(32, 512), nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.GroupNorm(32, 256), nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.GroupNorm(32, 128), nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.GroupNorm(32, 64), nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh(),
            )

    def forward(self, z):
        """z: [B, latent_dim] → список кадров [B, C, H, W] × n_frames.
        Никаких 5D на устройстве — только списки 4D тензоров."""
        b = z.size(0)
        frames = []
        for t in range(self.n_frames):
            t_vec = torch.zeros(b, self.n_frames, device=z.device)
            t_vec[:, t] = 1.0
            z_t = torch.cat([z, t_vec], dim=1)       # [B, L+T] — 2D
            lt = self.frame_mlp(z_t)                  # [B, L] — 2D
            lt = lt.view(b, self.latent_dim, 1, 1)    # [B, L, 1, 1] — 4D для conv
            frames.append(self.decoder(lt))           # [B, 3, H, W] — 4D
        return frames  # СПИСОК, не stack — нет 5D!


class VideoDiscriminator(nn.Module):
    """Дискриминатор для видео — совместим с DirectML.
    Принимает СПИСОК кадров [B, C, H, W] × T (не 5D!).
    Каждый кадр → conv (4D) → flatten (2D) → mean → MLP."""
    def __init__(self, image_size=64, n_frames=16):
        super().__init__()
        self.n_frames = n_frames
        feat_size = 512 if image_size == 64 else 1024

        if image_size == 64:
            self.frame_enc = nn.Sequential(
                spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(256, feat_size, 4, 2, 1)), nn.BatchNorm2d(feat_size), nn.LeakyReLU(0.2, True),
                nn.AdaptiveAvgPool2d(1),
            )
        else:
            self.frame_enc = nn.Sequential(
                spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
                spectral_norm(nn.Conv2d(512, feat_size, 4, 2, 1)), nn.BatchNorm2d(feat_size), nn.LeakyReLU(0.2, True),
                nn.AdaptiveAvgPool2d(1),
            )

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, frame_list):
        """frame_list: список [B, C, H, W] — НЕ 5D тензор!"""
        b = frame_list[0].size(0)
        feat_sum = None
        for frame in frame_list:               # каждый frame: [B, C, H, W] — 4D ✓
            f = self.frame_enc(frame)          # [B, feat, 1, 1] — 4D ✓
            f = f.view(b, -1)                  # [B, feat] — 2D ✓
            feat_sum = f if feat_sum is None else feat_sum + f
        feat_mean = feat_sum / len(frame_list) # [B, feat] — 2D ✓
        return self.classifier(feat_mean).view(-1)


class VGGPerceptualLoss(nn.Module):
    """Перцептивный loss на основе VGG-подобных feature extractor.
    Поддерживает:
      - автозагрузку torchvision VGG16
      - кастомные .pth/.pt/.safetensors модели из папки models_vgg/
    Работает на CPU если GPU/DirectML не тянет VGG."""
    def __init__(self, device, model_path: str = ""):
        super().__init__()
        self.device = device
        self.model_path = model_path  # "" = авто-VGG16
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        if self.model_path and os.path.isfile(self.model_path):
            # Кастомная модель из models_vgg/
            ext = os.path.splitext(self.model_path)[1].lower()
            if ext == ".safetensors":
                try:
                    from safetensors.torch import load_file as sf_load
                    weights = sf_load(self.model_path, device="cpu")
                except ImportError:
                    import subprocess, sys as _sys
                    subprocess.run([_sys.executable, "-m", "pip", "install", "safetensors"], check=True)
                    from safetensors.torch import load_file as sf_load
                    weights = sf_load(self.model_path, device="cpu")
            else:
                weights = torch.load(self.model_path, map_location="cpu", weights_only=False)
                if isinstance(weights, dict):
                    weights = weights.get("params", weights.get("model", weights))

            # Пробуем загрузить как VGG16 features
            try:
                from torchvision.models import vgg16
                vgg = vgg16(weights=None)
                vgg.features.load_state_dict(weights, strict=False)
                self._model = nn.Sequential(*list(vgg.features)[:16]).eval()
            except Exception:
                # Если не VGG — пробуем использовать как есть
                try:
                    from torchvision.models import vgg16
                    vgg = vgg16(weights=None)
                    self._model = nn.Sequential(*list(vgg.features)[:16]).eval()
                except Exception:
                    self._model = None
                    return
        else:
            # Авто — скачиваем VGG16 из torchvision
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            except Exception:
                from torchvision.models import vgg16
                vgg = vgg16(weights="DEFAULT")
            self._model = nn.Sequential(*list(vgg.features)[:16]).eval()

        for p in self._model.parameters():
            p.requires_grad_(False)

        # VGG нормализация (ImageNet)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        try:
            self._model = self._model.to(self.device)
            self._mean = self._mean.to(self.device)
            self._std  = self._std.to(self.device)
        except Exception:
            pass  # остаётся на CPU

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        self._load()
        if self._model is None:
            return fake.new_tensor(0.0)

        def prep(x):
            x = (x.float() * 0.5 + 0.5).clamp(0, 1)
            return (x - self._mean) / self._std

        try:
            f_fake = self._model(prep(fake.to(self.device)))
            f_real = self._model(prep(real.to(self.device)))
            return F.l1_loss(f_fake, f_real.detach())
        except Exception:
            # Fallback CPU
            try:
                cpu = self._model.cpu()
                m = self._mean.cpu(); s = self._std.cpu()
                def prep_cpu(x): return ((x.cpu().float()*0.5+0.5).clamp(0,1)-m)/s
                return F.l1_loss(cpu(prep_cpu(fake)), cpu(prep_cpu(real)).detach())
            except Exception:
                return fake.new_tensor(0.0)


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif "GroupNorm" in name:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class FlatImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # GAN does not need real class labels here.
        return image, 0



# ── Звуковое уведомление (без внешних зависимостей) ──────────────────────────
def _play_done_sound():
    """Генерирует простой beep через wave модуль — работает без winsound."""
    if not SETTINGS.get("enable_sound_notify"):
        return
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
        return
    except Exception:
        pass
    try:
        # Генерируем WAV в памяти и воспроизводим
        duration, freq, rate = 0.3, 880, 44100
        n = int(rate * duration)
        buf = struct.pack("<" + "h" * n,
                         *[int(32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)])
        with io.BytesIO() as f:
            with wave.open(f, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(buf)
            f.seek(0)
            data = f.read()
        tmp = os.path.join(OUTPUT_DIR, "_beep_tmp.wav")
        with open(tmp, "wb") as f2:
            f2.write(data)
        import subprocess
        subprocess.Popen(["powershell", "-c", f"(New-Object Media.SoundPlayer '{tmp}').PlaySync()"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── Окно настроек ──────────────────────────────────────────────────────────────
class SettingsWindow:
    def __init__(self, root):
        self.root = root
        self.dlg = tk.Toplevel(root)
        self.dlg.title("⚙ Настройки")
        self.dlg.geometry("640x680")
        self.dlg.configure(bg="#141821")
        self.dlg.transient(root)
        self.dlg.grab_set()
        self._vars = {}
        self._build()

    def _build(self):
        style = ttk.Style()

        canvas = tk.Canvas(self.dlg, bg="#141821", highlightthickness=0)
        sb = ttk.Scrollbar(self.dlg, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        sb.pack(side="right", fill="y", pady=8)

        def section(parent, title):
            f = ttk.LabelFrame(parent, text=title, padding=8)
            f.pack(fill="x", padx=8, pady=4)
            return f

        def toggle(parent, key, label, hint=""):
            var = tk.BooleanVar(value=bool(SETTINGS.get(key)))
            self._vars[key] = var
            row = ttk.Frame(parent)
            row.pack(anchor="w", pady=2)
            ttk.Checkbutton(row, text=label, variable=var).pack(side="left")
            if hint:
                ttk.Label(row, text=f"  ({hint})", style="Hint.TLabel").pack(side="left")

        def spinval(parent, key, label, from_, to, hint=""):
            var = tk.IntVar(value=int(SETTINGS.get(key, from_)))
            self._vars[key] = var
            row = ttk.Frame(parent)
            row.pack(anchor="w", pady=2)
            ttk.Label(row, text=label, width=28).pack(side="left")
            ttk.Spinbox(row, from_=from_, to=to, textvariable=var, width=8).pack(side="left", padx=4)
            if hint:
                ttk.Label(row, text=hint, style="Hint.TLabel").pack(side="left", padx=4)

        def combobox(parent, key, label, values):
            var = tk.StringVar(value=str(SETTINGS.get(key, values[0])))
            self._vars[key] = var
            row = ttk.Frame(parent)
            row.pack(anchor="w", pady=2)
            ttk.Label(row, text=label, width=28).pack(side="left")
            ttk.Combobox(row, values=values, textvariable=var, state="readonly", width=14).pack(side="left", padx=4)

        # ── ОБУЧЕНИЕ ──
        f = section(sf, "🧠 Обучение")
        toggle(f, "enable_grad_accum", "Gradient Accumulation", "увеличивает эффективный batch")
        spinval(f, "grad_accum_steps", "Шагов накопления:", 1, 32)
        toggle(f, "enable_fp16", "FP16 (half precision)", "ускорение ~1.5x, нестабильно на DirectML")
        toggle(f, "enable_multi_gpu", "Multi-GPU (DataParallel)", "для систем с несколькими картами")
        toggle(f, "enable_fid_score", "FID Score метрика", "требует scipy — ставится автоматически")
        spinval(f, "fid_every_n_epochs", "FID каждые N эпох:", 1, 50)
        toggle(f, "enable_early_stopping", "Early Stopping", "остановка при отсутствии прогресса")
        spinval(f, "early_stopping_patience", "Терпение (эпох):", 1, 100)
        combobox(f, "lr_scheduler", "Планировщик LR:", ["cosine", "step", "none"])
        spinval(f, "lr_warmup_epochs", "Warmup эпох:", 0, 20)

        # ── ДАТАСЕТ ──
        f = section(sf, "🗂 Датасет")
        toggle(f, "enable_phash_dedup", "pHash дедупликация", "убирает похожие изображения")
        spinval(f, "phash_threshold", "Порог схожести pHash:", 1, 32, "меньше = строже")
        toggle(f, "enable_clip_filter", "CLIP фильтрация", "удаляет низкокачественные изображения")
        spinval(f, "clip_min_score", "Мин. CLIP score (x100):", 1, 99)
        toggle(f, "enable_face_crop", "Автокадрирование лиц", "нужен opencv-python")
        toggle(f, "enable_balance_dataset", "Балансировка датасета", "выравнивает портреты/тела/позы")

        # ── ИНТЕРФЕЙС ──
        f = section(sf, "🎨 Интерфейс")
        combobox(f, "theme", "Тема:", ["dark", "light"])
        combobox(f, "language", "Язык:", ["ru", "en"])
        toggle(f, "enable_sound_notify", "Звук по завершению", "beep когда обучение закончилось")
        toggle(f, "enable_toast_notify", "Toast уведомление", "всплывающее окно")
        toggle(f, "enable_file_log", "Лог в файл", f"training_log.txt рядом с программой")

        # ── ГЕНЕРАЦИЯ ──
        f = section(sf, "🖼 Генерация")
        toggle(f, "enable_onnx_export", "ONNX экспорт", "экспорт модели для других программ")

        # ── Кнопки ──
        btn_f = ttk.Frame(sf)
        btn_f.pack(fill="x", padx=8, pady=(12, 8))
        ttk.Button(btn_f, text="💾 Сохранить", command=self._save,
                   style="Primary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(btn_f, text="↺ По умолчанию", command=self._reset).pack(side="left", padx=(0, 8))
        ttk.Button(btn_f, text="Закрыть", command=self.dlg.destroy).pack(side="right")

    def _save(self):
        for key, var in self._vars.items():
            val = var.get()
            # IntVar может быть float из spinbox
            if key in DEFAULT_SETTINGS and isinstance(DEFAULT_SETTINGS[key], bool):
                val = bool(val)
            elif key in DEFAULT_SETTINGS and isinstance(DEFAULT_SETTINGS[key], int):
                try:
                    val = int(val)
                except Exception:
                    pass
            SETTINGS.set(key, val)
        SETTINGS.save()
        messagebox.showinfo("Настройки", "Настройки сохранены!\nНекоторые изменения вступят в силу при следующем запуске.")
        self.dlg.destroy()

    def _reset(self):
        for key, var in self._vars.items():
            default = DEFAULT_SETTINGS.get(key)
            if default is not None:
                var.set(default)


class ImageGANApp:
    def __init__(self, root):
        self.root = root
        self.lang = SETTINGS.get("language", "ru")
        self.root.title("AI Image Generator (GAN) — Copyright (C) 2026 Ivan Nedostup (GGB_638)")
        self.root.geometry("980x720")

        ensure_dirs()
        self.device, self.device_name = pick_device()
        self.latest_checkpoint = ""
        self.busy = False
        self._stop_requested = False
        self.status_var = tk.StringVar(value=self.msg("status_ready"))
        self._activity_tick = 0
        self._activity_after_id = None
        self._current_run_losses_d = []
        self._current_run_losses_g = []
        self._run_start_epoch = 0

        self._build_ui()
        self._apply_theme()
        self._setup_hotkeys()
        self._setup_drag_drop()
        self.refresh_last_training_preview()
        self.refresh_generated_preview()
        self.apply_training_profile(initial=True)
        self.log("AI Image Generator (GAN)")
        self.log("Copyright (C) 2026 Ivan Nidostup (GGB_638). Distributed under GNU GPL v3.")
        self.log("This program comes with ABSOLUTELY NO WARRANTY; see LICENSE file for details.")
        self.log(f"Device: {self.device_name}")
        self.log("Ready.")
        log_to_file(f"App started. Device: {self.device_name}")

    def msg(self, key: str, **kwargs) -> str:
        text = LANG_MESSAGES.get(self.lang, LANG_MESSAGES["ru"]).get(key, key)
        return text.format(**kwargs) if kwargs else text

    def set_language(self):
        selected = self.lang_var.get()
        self.lang = "en" if selected == "English" else "ru"
        self.set_status(self.msg("status_ready"))
        self.log(self.msg("lang_switched"))

    def _is_main_thread(self) -> bool:
        return threading.current_thread() is threading.main_thread()

    def _ui_call(self, fn, *args, **kwargs):
        if self._is_main_thread():
            fn(*args, **kwargs)
        else:
            self.root.after(0, lambda: fn(*args, **kwargs))

    def set_status(self, text: str):
        self._ui_call(self.status_var.set, text)

    def show_info(self, title: str, message: str):
        self._ui_call(messagebox.showinfo, title, message)

    def show_warning(self, title: str, message: str):
        self._ui_call(messagebox.showwarning, title, message)

    def show_error(self, title: str, message: str):
        self._ui_call(messagebox.showerror, title, message)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="AI Image Generator", style="Title.TLabel")
        title.pack(anchor="w", pady=(0, 2))

        subtitle = ttk.Label(
            main,
            text="Для новичков: 1) Скачай картинки -> 2) Обучи модель -> 3) Сгенерируй результат",
            style="Hint.TLabel",
        )
        subtitle.pack(anchor="w", pady=(0, 2))

        copyright_label = ttk.Label(
            main,
            text="Copyright (C) 2026 Ivan Nidostup (GGB_638). Distributed under GNU GPL v3. | No warranty.",
            style="Hint.TLabel",
        )
        copyright_label.pack(anchor="w", pady=(0, 6))

        top_info = ttk.Label(main, text=f"Рабочие папки: {DATASET_DIR}/  {CHECKPOINT_DIR}/  {OUTPUT_DIR}/", style="Hint.TLabel")
        top_info.pack(anchor="w", pady=(0, 8))

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=(0, 8))
        ttk.Label(actions, text="Language:").pack(side="left", padx=(0, 6))
        self.lang_var = tk.StringVar(value="Русский")
        ttk.Combobox(actions, values=["Русский", "English"], textvariable=self.lang_var, state="readonly", width=12).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Apply", command=self.set_language).pack(side="left", padx=(0, 10))
        ttk.Button(actions, text="Проверить готовность", command=self.check_readiness).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Быстрый старт обучения", command=self.quick_start_training, style="Primary.TButton").pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Полный авто-режим", command=self.full_auto_mode, style="Primary.TButton").pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Прогрессивно 64→128→256", command=self.progressive_auto_mode).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Анти-баг проверка", command=self.run_diagnostics).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="⚙ Настройки", command=self.open_settings).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="🔄 Обновления", command=self.check_for_updates).pack(
            side="left", padx=(0, 6)
        )
        ttk.Label(actions, textvariable=self.status_var, style="Hint.TLabel").pack(side="left", padx=(8, 0))
        self.busy_bar = ttk.Progressbar(actions, mode="indeterminate", length=130)
        self.busy_bar.pack(side="left", padx=(8, 0))

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True, pady=(0, 0))

        self.tab_download = ttk.Frame(notebook, padding=10)
        self.tab_train = ttk.Frame(notebook, padding=10)
        self.tab_generate = ttk.Frame(notebook, padding=10)
        self.tab_video = ttk.Frame(notebook, padding=10)
        self.tab_gallery = ttk.Frame(notebook, padding=10)
        self.tab_history = ttk.Frame(notebook, padding=10)
        self.tab_settings = ttk.Frame(notebook, padding=10)

        notebook.add(self.tab_download, text="Скачивание")
        notebook.add(self.tab_train, text="Обучение")
        notebook.add(self.tab_generate, text="Генерация")
        notebook.add(self.tab_video, text="🎬 Видео")
        notebook.add(self.tab_gallery, text="🖼 Галерея")
        notebook.add(self.tab_history, text="📈 История")
        notebook.add(self.tab_settings, text="⚙ Настройки")

        self._build_download_tab()
        self._build_train_tab()
        self._build_generate_tab()
        self._build_video_tab()
        self._build_gallery_tab()
        self._build_history_tab()
        self._build_settings_tab()

        # Лог — всегда внизу, фиксированная высота, не перекрывается notebook
        log_frame = ttk.LabelFrame(main, text="Лог  (F2 — очистить)")
        log_frame.pack(side="bottom", fill="x", expand=False, pady=(4, 0))
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical")
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(log_frame, height=7, state="disabled",
                                yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="x", expand=True)
        log_scroll.configure(command=self.log_text.yview)
        self.log_text.configure(bg="#0F1320", fg="#D5DEF5",
                                insertbackground="#D5DEF5", selectbackground="#2B3550")
        # F2 — очистить лог
        self.root.bind("<F2>", lambda e: (
            self.log_text.configure(state="normal"),
            self.log_text.delete("1.0", "end"),
            self.log_text.configure(state="disabled")
        ))

    def _build_download_tab(self):
        ttk.Label(self.tab_download,
                  text="Шаг 1: Выберите режим, вставьте URL и нажмите Скачать.",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # Переключатель режима скачивания
        mode_frame = ttk.LabelFrame(self.tab_download, text="Что скачивать", padding=6)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.download_mode_var = tk.StringVar(value="images")
        ttk.Radiobutton(mode_frame, text="🖼 Картинки  → dataset/",
                        variable=self.download_mode_var, value="images",
                        command=self._on_download_mode_change).pack(side="left", padx=8)
        ttk.Radiobutton(mode_frame, text="🎬 Видео  → dataset_video/",
                        variable=self.download_mode_var, value="video",
                        command=self._on_download_mode_change).pack(side="left", padx=8)
        self.download_mode_hint = ttk.Label(mode_frame,
                                             text=f"Папка: {DATASET_DIR}", style="Hint.TLabel")
        self.download_mode_hint.pack(side="left", padx=12)

        ttk.Label(self.tab_download, text="URL сайта:").grid(row=2, column=0, sticky="w", pady=4)
        self.url_var = tk.StringVar(value="https://unsplash.com")
        ttk.Entry(self.tab_download, textvariable=self.url_var, width=70).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=4)

        ttk.Label(self.tab_download, text="Сколько файлов:").grid(row=3, column=0, sticky="w", pady=4)
        self.count_var = tk.IntVar(value=100)
        ttk.Spinbox(self.tab_download, from_=10, to=5000, textvariable=self.count_var, width=12).grid(
            row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(self.tab_download, text="Фильтр качества:").grid(row=4, column=0, sticky="w", pady=4)
        self.download_filter_var = tk.StringVar(value="strict")
        ttk.Combobox(self.tab_download, values=["strict", "normal"], state="readonly",
                     width=12, textvariable=self.download_filter_var).grid(
            row=4, column=1, sticky="w", padx=6, pady=4)

        btn_f = ttk.Frame(self.tab_download)
        btn_f.grid(row=5, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(btn_f, text="Скачать", command=self.start_download,
                   style="Primary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(btn_f, text="Очистить dataset/", command=self.start_cleanup_dataset).pack(side="left", padx=(0, 8))
        ttk.Button(btn_f, text="Очистить dataset_video/",
                   command=lambda: self.run_async(self._cleanup_video_dataset)).pack(side="left")

        # Тематические подпапки датасета
        sub_frame = ttk.LabelFrame(self.tab_download, text="Тематические подпапки  (для разных стилей)", padding=6)
        sub_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(sub_frame, text="Скачать в подпапку:").pack(side="left")
        self.subfolder_var = tk.StringVar(value="")
        ttk.Entry(sub_frame, textvariable=self.subfolder_var, width=20).pack(side="left", padx=6)
        ttk.Label(sub_frame, text="(пусто = в корень dataset/)", style="Hint.TLabel").pack(side="left")
        ttk.Button(sub_frame, text="📁 Создать папку",
                   command=self._create_subfolder).pack(side="left", padx=(8, 0))
        self._refresh_subfolder_list(sub_frame)

        # Счётчик файлов в обоих датасетах
        self.dataset_info_var = tk.StringVar(value="")
        ttk.Label(self.tab_download, textvariable=self.dataset_info_var, style="Hint.TLabel").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=4)
        self.tab_download.columnconfigure(1, weight=1)
        self.tab_download.after(500, self._refresh_dataset_info)

    def _on_download_mode_change(self):
        mode = self.download_mode_var.get()
        folder = DATASET_DIR if mode == "images" else DATASET_VIDEO_DIR
        self.download_mode_hint.configure(text=f"Папка: {folder}")

    def _create_subfolder(self):
        name = self.subfolder_var.get().strip()
        if not name:
            self.show_info("Подпапка", "Введи название подпапки.")
            return
        path = os.path.join(DATASET_DIR, name)
        os.makedirs(path, exist_ok=True)
        self.log(f"Создана подпапка: dataset/{name}/")
        self._refresh_dataset_info()

    def _refresh_subfolder_list(self, parent_frame=None):
        """Показывает список существующих подпапок датасета."""
        if not os.path.isdir(DATASET_DIR):
            return
        subs = [d for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if subs and parent_frame:
            lbl_text = "Подпапки: " + ", ".join(subs)
            if hasattr(self, "_subfolder_hint_label"):
                self._subfolder_hint_label.configure(text=lbl_text)
            else:
                self._subfolder_hint_label = ttk.Label(
                    parent_frame, text=lbl_text, style="Hint.TLabel")
                self._subfolder_hint_label.pack(side="left", padx=(8, 0))

    def _refresh_dataset_info(self):
        img_count = self.count_dataset_images()
        vid_count = sum(1 for f in os.listdir(DATASET_VIDEO_DIR)
                        if f.lower().endswith(ALLOWED_VIDEO_EXTENSIONS)) if os.path.isdir(DATASET_VIDEO_DIR) else 0
        self.dataset_info_var.set(
            f"dataset/ (картинки): {img_count} файлов    |    dataset_video/ (видео): {vid_count} файлов"
        )

    def _cleanup_video_dataset(self):
        if not os.path.isdir(DATASET_VIDEO_DIR):
            return
        removed = 0
        for f in os.listdir(DATASET_VIDEO_DIR):
            try:
                os.remove(os.path.join(DATASET_VIDEO_DIR, f))
                removed += 1
            except Exception:
                pass
        self.log(f"Очищено dataset_video/: удалено {removed} файлов")
        self._ui_call(self._refresh_dataset_info)

    def _build_train_tab(self):
        # Создаем Canvas с прокруткой для вкладки обучения
        canvas = tk.Canvas(self.tab_train, bg="#141821", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_train, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=10)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Прокрутка колесом мыши
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _bind_mousewheel_to_all(widget):
            """Привязать прокрутку ко всем виджетам внутри фрейма"""
            widget.bind("<MouseWheel>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel_to_all(child)
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        # Привязываем ко всем дочерним виджетам после создания содержимого
        scrollable_frame.after(100, lambda: _bind_mousewheel_to_all(scrollable_frame))

        help_text = (
            "Шаг 2: Выберите профиль и нажмите Начать обучение.\n"
            "Новичкам лучше профиль Авто + DiffAugment включено."
        )
        ttk.Label(scrollable_frame, text=help_text, style="Hint.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(scrollable_frame, text="Профиль:").grid(row=1, column=0, sticky="w", pady=4)
        self.profile_var = tk.StringVar(value="Авто (рекомендуется)")
        profile_combo = ttk.Combobox(
            scrollable_frame,
            values=["Авто (рекомендуется)", "Быстро", "Баланс", "Качество"],
            state="readonly",
            width=24,
            textvariable=self.profile_var,
        )
        profile_combo.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        profile_combo.current(0)
        ttk.Button(scrollable_frame, text="Применить профиль", command=self.apply_training_profile).grid(
            row=1, column=2, sticky="w", padx=6, pady=4
        )

        ttk.Label(scrollable_frame, text="DiffAugment:").grid(row=2, column=0, sticky="w", pady=4)
        self.diffaug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, variable=self.diffaug_var, text="Включено").grid(
            row=2, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(scrollable_frame, text="Эпохи:").grid(row=3, column=0, sticky="w", pady=4)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(scrollable_frame, from_=1, to=1000, textvariable=self.epochs_var, width=12).grid(
            row=3, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(scrollable_frame, text="Batch size:").grid(row=4, column=0, sticky="w", pady=4)
        self.batch_var = tk.IntVar(value=16)
        ttk.Spinbox(scrollable_frame, from_=1, to=256, textvariable=self.batch_var, width=12).grid(
            row=4, column=1, sticky="w", padx=6, pady=4
        )

        ttk.Label(scrollable_frame, text="Размер изображения:").grid(row=5, column=0, sticky="w", pady=4)
        self.size_var = tk.IntVar(value=64)
        size_combo = ttk.Combobox(scrollable_frame, values=[64, 128, 256], state="readonly", width=10, textvariable=self.size_var)
        size_combo.grid(row=5, column=1, sticky="w", padx=6, pady=4)
        size_combo.current(0)

        ttk.Button(scrollable_frame, text="Начать обучение", command=self.start_training, style="Primary.TButton").grid(
            row=6, column=1, sticky="w", padx=6, pady=8
        )
        ttk.Button(scrollable_frame, text="Дообучить на новом размере", command=self.start_upscale_finetune).grid(
            row=6, column=0, sticky="w", padx=6, pady=8
        )
        ttk.Button(scrollable_frame, text="Дообучить", command=self.start_resume_training).grid(
            row=6, column=2, sticky="w", padx=6, pady=8
        )
        ttk.Button(scrollable_frame, text="Прогрессивно (кастом)", command=self.progressive_custom_mode).grid(
            row=9, column=2, sticky="w", padx=6, pady=8
        )

        ttk.Label(scrollable_frame, text="Чекпоинт для дообучения:").grid(row=7, column=0, sticky="w", pady=4)
        self.resume_ckpt_var = tk.StringVar(value="")
        ttk.Entry(scrollable_frame, textvariable=self.resume_ckpt_var, width=60).grid(row=7, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(scrollable_frame, text="Последний", command=self.fill_latest_resume_checkpoint).grid(
            row=7, column=2, sticky="w", padx=6, pady=4
        )
        ttk.Label(scrollable_frame, text="Последний этап обучения:").grid(row=8, column=0, sticky="nw", pady=(10, 4))
        self.train_preview_label = ttk.Label(scrollable_frame, text="Нет превью")
        self.train_preview_label.grid(row=8, column=1, columnspan=2, sticky="w", padx=6, pady=(10, 4))

        ttk.Label(scrollable_frame, text="Кастом прогрессия (epochs / batch):").grid(row=9, column=0, sticky="w", pady=(8, 4))
        cfg = ttk.Frame(scrollable_frame)
        cfg.grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 8))
        ttk.Label(cfg, text="64:").grid(row=0, column=0, sticky="w")
        self.prog_ep64_var = tk.IntVar(value=24)
        self.prog_bs64_var = tk.IntVar(value=16)
        ttk.Spinbox(cfg, from_=1, to=1000, textvariable=self.prog_ep64_var, width=8).grid(row=0, column=1, padx=4)
        ttk.Spinbox(cfg, from_=1, to=256, textvariable=self.prog_bs64_var, width=8).grid(row=0, column=2, padx=4)
        ttk.Label(cfg, text="128:").grid(row=0, column=3, sticky="w", padx=(10, 0))
        self.prog_ep128_var = tk.IntVar(value=16)
        self.prog_bs128_var = tk.IntVar(value=8)
        ttk.Spinbox(cfg, from_=1, to=1000, textvariable=self.prog_ep128_var, width=8).grid(row=0, column=4, padx=4)
        ttk.Spinbox(cfg, from_=1, to=256, textvariable=self.prog_bs128_var, width=8).grid(row=0, column=5, padx=4)
        ttk.Label(cfg, text="256:").grid(row=0, column=6, sticky="w", padx=(10, 0))
        self.prog_ep256_var = tk.IntVar(value=10)
        self.prog_bs256_var = tk.IntVar(value=4)
        ttk.Spinbox(cfg, from_=1, to=1000, textvariable=self.prog_ep256_var, width=8).grid(row=0, column=7, padx=4)
        ttk.Spinbox(cfg, from_=1, to=256, textvariable=self.prog_bs256_var, width=8).grid(row=0, column=8, padx=4)
        ttk.Label(scrollable_frame, text="EMA decay:").grid(row=11, column=0, sticky="w", pady=4)
        self.ema_decay_var = tk.DoubleVar(value=0.995)
        ttk.Spinbox(scrollable_frame, from_=0.9, to=0.9999, increment=0.001,
                    textvariable=self.ema_decay_var, width=12, format="%.4f").grid(
            row=11, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(scrollable_frame, text="(0.990–0.999, чем выше — тем плавнее EMA)", style="Hint.TLabel").grid(
            row=11, column=2, sticky="w", padx=4)

        # Перцептивный loss
        perc_frame = ttk.LabelFrame(scrollable_frame, text="Перцептивный loss (VGG features)", padding=6)
        perc_frame.grid(row=13, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.perceptual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(perc_frame, variable=self.perceptual_var,
                        text="Включить  (генератор учится по 'смыслу', а не попиксельно)").grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(perc_frame, text="Модель VGG:").grid(row=1, column=0, sticky="w", pady=4)
        self.vgg_model_var = tk.StringVar(value="Авто (VGG16 torchvision ~528MB)")
        self.vgg_model_combo = ttk.Combobox(perc_frame, textvariable=self.vgg_model_var,
                                             state="readonly", width=42)
        self.vgg_model_combo.grid(row=1, column=1, sticky="w", padx=4)
        ttk.Button(perc_frame, text="🔄", width=3,
                   command=self._refresh_vgg_model_list).grid(row=1, column=2, padx=4)
        ttk.Label(perc_frame,
                  text=f"Папка: models_vgg/  |  Форматы: .pth  .pt  .safetensors  .ckpt",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=3, sticky="w")

        # Прогресс-бар с реальным временем
        ttk.Label(scrollable_frame, text="Прогресс:").grid(row=14, column=0, sticky="w", pady=4)
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.train_progress_bar = ttk.Progressbar(scrollable_frame, variable=self.train_progress_var,
                                                   maximum=100, length=300)
        self.train_progress_bar.grid(row=14, column=1, sticky="w", padx=6)
        self.train_progress_label = ttk.Label(scrollable_frame, text="—", style="Hint.TLabel")
        self.train_progress_label.grid(row=14, column=2, sticky="w", padx=4)
        scrollable_frame.after(200, self._refresh_vgg_model_list)

        # --- График Loss D/G в реальном времени ---
        ttk.Label(scrollable_frame, text="График обучения:").grid(row=12, column=0, sticky="nw", pady=(10, 4))
        self.loss_canvas = tk.Canvas(scrollable_frame, width=500, height=180,
                                     bg="#0F1320", highlightthickness=1, highlightbackground="#2B3550")
        self.loss_canvas.grid(row=12, column=1, columnspan=2, sticky="w", padx=6, pady=(10, 4))
        self._loss_d_history = []
        self._loss_g_history = []
        self._draw_loss_graph()

        scrollable_frame.columnconfigure(1, weight=1)

    def apply_training_profile(self, initial=False):
        profile = self.profile_var.get().strip()
        if profile == "Авто (рекомендуется)":
            if "CPU" in self.device_name:
                epochs, batch, size = 30, 4, 64
            elif "DirectML" in self.device_name:
                epochs, batch, size = 50, 8, 64
            else:
                epochs, batch, size = 80, 16, 128
        elif profile == "Быстро":
            epochs, batch, size = 20, 8, 64
        elif profile == "Качество":
            if "CUDA" in self.device_name:
                epochs, batch, size = 140, 8, 256
            else:
                epochs, batch, size = 120, 8, 128
        else:
            epochs, batch, size = 60, 8, 64

        self.epochs_var.set(epochs)
        self.batch_var.set(batch)
        self.size_var.set(size)
        if not initial:
            self.log(f"Profile applied: {profile} -> epochs={epochs}, batch={batch}, size={size}")

    def count_dataset_images(self):
        return len(
            [
                f
                for f in os.listdir(DATASET_DIR)
                if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)
            ]
        )

    def newest_checkpoint_path(self):
        ckpts = sorted(
            [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
            key=os.path.getmtime,
        )
        return ckpts[-1] if ckpts else ""

    def fill_latest_resume_checkpoint(self):
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Чекпоинты не найдены в checkpoints/.")
        self.resume_ckpt_var.set(ckpt)
        # Автоматически определяем размер из чекпоинта
        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            ckpt_size = int(state.get("image_size", 64))
            if self.size_var.get() != ckpt_size:
                self.log(f"Auto-adjusted size: {self.size_var.get()} -> {ckpt_size} (from checkpoint)")
                self.size_var.set(ckpt_size)
        except Exception:
            pass
        self.log(f"Resume checkpoint selected: {ckpt}")

    def _load_partial_state(self, module: nn.Module, state_dict: dict) -> int:
        """Load only matching keys/shapes for cross-resolution fine-tuning."""
        own = module.state_dict()
        matched = {}
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                matched[k] = v
        if matched:
            own.update(matched)
            module.load_state_dict(own)
        return len(matched)

    def check_readiness(self, show_popup=True):
        image_count = self.count_dataset_images()
        rec = []
        rec.append(f"Устройство: {self.device_name}")
        rec.append(f"Картинок в dataset/: {image_count}")
        rec.append(f"Текущий профиль: {self.profile_var.get().strip()}")
        rec.append(
            f"Параметры: epochs={self.epochs_var.get()}, batch={self.batch_var.get()}, size={self.size_var.get()}, "
            f"DiffAugment={'ON' if self.diffaug_var.get() else 'OFF'}"
        )

        if image_count < 50:
            rec.append("Рекомендация: мало данных. Лучше скачать минимум 100-300 картинок.")
            self.set_status(self.msg("status_low_data"))
        elif image_count < 200:
            rec.append("Рекомендация: можно обучать, но качество вырастет при 300+ картинках.")
            self.set_status(self.msg("status_can_train"))
        else:
            rec.append("Готово к обучению.")
            self.set_status(self.msg("status_ready_train"))

        self.log(" | ".join(rec))
        if show_popup:
            self.show_info(self.msg("readiness_title"), "\n".join(rec))

    def run_diagnostics(self):
        """Проверка системы на ошибки и баги"""
        self.run_async(self._diagnostics_flow)

    def _diagnostics_flow(self):
        results = []
        errors = []
        warnings = []
        self.log("=== ПОЛНАЯ АНТИ-БАГ ПРОВЕРКА ===")

        # 1. Устройство
        try:
            results.append(f"✓ Устройство: {self.device_name}")
            if "CPU" in self.device_name:
                warnings.append("⚠ CPU — обучение очень медленное. Рекомендуется GPU.")
            elif "CUDA" in self.device_name and torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - torch.cuda.memory_allocated() / 1024**3
                results.append(f"  GPU: {name}  |  VRAM: {total:.1f} GB  |  Свободно: {free:.1f} GB")
                if total < 4: warnings.append("⚠ VRAM <4GB — только 64px batch=4-8")
                elif total < 8: warnings.append("⚠ VRAM <8GB — 256px может не влезть")
            elif "DirectML" in self.device_name:
                warnings.append("⚠ DirectML — работает, но медленнее CUDA. Нормально для AMD/Intel.")
        except Exception as e:
            errors.append(f"✗ Устройство: {e}")

        # 2. Папки
        for folder, name in [
            (DATASET_DIR,"dataset"),(DATASET_VIDEO_DIR,"dataset_video"),
            (CHECKPOINT_DIR,"checkpoints"),(OUTPUT_DIR,"output"),
            (MODELS_DIR,"models"),(VGG_MODELS_DIR,"models_vgg")
        ]:
            if os.path.isdir(folder):
                results.append(f"✓ {name}/  существует")
            else:
                try:
                    os.makedirs(folder, exist_ok=True)
                    results.append(f"✓ {name}/  создана")
                except Exception as e:
                    errors.append(f"✗ {name}/  не создать: {e}")

        # 3. Датасет
        image_count = self.count_dataset_images()
        if image_count == 0:
            errors.append("✗ dataset/ пуст — скачай изображения")
        elif image_count < 100:
            warnings.append(f"⚠ Мало изображений: {image_count} (идеально 1000+)")
        else:
            results.append(f"✓ Датасет: {image_count} изображений")

        broken = 0
        for fname in os.listdir(DATASET_DIR):
            if fname.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                try:
                    Image.open(os.path.join(DATASET_DIR, fname)).verify()
                except Exception:
                    broken += 1
        if broken:
            warnings.append(f"⚠ Битых файлов в датасете: {broken} — запусти Очистить dataset")
        else:
            results.append("✓ Битых изображений не найдено")

        # 4. Чекпоинты + тест загрузки модели
        ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        if ckpts:
            latest = self.newest_checkpoint_path()
            size_mb = os.path.getsize(latest) / 1024**2
            results.append(f"✓ Чекпоинтов: {len(ckpts)} (BEST: {sum(1 for f in ckpts if 'BEST' in f)})")
            try:
                state = torch.load(latest, map_location="cpu", weights_only=False)
                img_size = int(state.get("image_size", 64))
                ld = int(state.get("latent_dim", LATENT_DIM))
                results.append(f"  Последний: {os.path.basename(latest)} ({size_mb:.0f}MB, {img_size}px)")
                # Реальный тест — загружаем и прогоняем forward
                g = Generator(image_size=img_size, latent_dim=ld)
                gen_state = state.get("generator_ema_state_dict", state.get("generator_state_dict", {}))
                g.load_state_dict(gen_state, strict=False)
                g.eval()
                with torch.no_grad():
                    dummy = torch.randn(1, ld, 1, 1)
                    out = g(dummy)
                    results.append(f"  Тест forward: вход [1,{ld},1,1] → выход {list(out.shape)} ✓")
            except Exception as e:
                errors.append(f"✗ Чекпоинт повреждён или несовместим: {e}")
        else:
            warnings.append("⚠ Нет чекпоинтов — нужно обучить модель")

        # 5. Зависимости — проверяем импорт И версию
        deps = [
            ("torch","PyTorch","2.0"),("torchvision","torchvision","0.15"),
            ("PIL","Pillow","9.0"),("requests","requests","2.0"),("bs4","BeautifulSoup",""),
        ]
        for mod, name, min_ver in deps:
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "?")
                results.append(f"✓ {name}: {ver}")
            except ImportError:
                errors.append(f"✗ {name} не установлен → pip install {mod}")

        # DirectML
        if "DirectML" in self.device_name:
            try:
                import torch_directml
                results.append("✓ torch-directml: установлен")
            except ImportError:
                errors.append("✗ torch-directml не найден → pip install torch-directml")

        # 6. Реальный тест обучающей итерации (1 шаг без GPU)
        try:
            size_test = 64
            g_test = Generator(image_size=size_test, latent_dim=10)
            d_test = Discriminator(image_size=size_test)
            z = torch.randn(2, 10, 1, 1)
            fake = g_test(z)
            score = d_test(fake)
            loss = score.mean()
            loss.backward()
            results.append(f"✓ Тест G→D→backward: OK (CPU, latent=10, size=64)")
        except Exception as e:
            errors.append(f"✗ Тест G→D провалился: {e}")

        # 7. Тест диффеяугментации
        try:
            test_img = torch.randn(2, 3, 64, 64)
            aug = diff_augment(test_img)
            assert aug.shape == test_img.shape
            results.append("✓ DiffAugment: OK")
        except Exception as e:
            errors.append(f"✗ DiffAugment: {e}")

        # 8. Параметры обучения
        epochs = self.epochs_var.get()
        batch = self.batch_var.get()
        size = self.size_var.get()
        if epochs < 15: warnings.append(f"⚠ Мало эпох ({epochs}) — рекомендуется 50+")
        else: results.append(f"✓ Эпохи: {epochs}")
        if batch < 1: errors.append(f"✗ Batch size = {batch} — некорректно")
        elif batch > 32 and "DirectML" in self.device_name:
            warnings.append(f"⚠ Batch={batch} для DirectML рискованно — попробуй 8-16")
        else: results.append(f"✓ Batch: {batch},  Размер: {size}px")

        # 9. Объяснение про 64px
        if size == 64:
            warnings.append(
                "ℹ 64px — это потолок качества. Лица будут узнаваемы но всегда немного 'мыльными'.\n"
                "  Для резкости переходи на 128px (нужно больше VRAM и времени).\n"
                "  Используй Feature Matching + перцептивный loss для лучших деталей на 64px."
            )

        # 10. Проверка записи в OUTPUT_DIR
        try:
            test_f = os.path.join(OUTPUT_DIR, "_test_write.tmp")
            with open(test_f, "w") as f: f.write("test")
            os.remove(test_f)
            results.append("✓ Запись в output/: OK")
        except Exception as e:
            errors.append(f"✗ Нет прав записи в output/: {e}")

        # Итог
        report_lines = [
            "=" * 52,
            f"АНТИ-БАГ ПРОВЕРКА  |  Ошибок: {len(errors)}  |  Предупреждений: {len(warnings)}",
            "=" * 52, "",
        ]
        report_lines.extend(results)
        if warnings:
            report_lines += ["", "─── ПРЕДУПРЕЖДЕНИЯ / СОВЕТЫ ───"]
            report_lines.extend(warnings)
        if errors:
            report_lines += ["", "─── КРИТИЧЕСКИЕ ОШИБКИ ───"]
            report_lines.extend(errors)
            report_lines += ["", "⛔ Исправь ошибки перед началом работы!"]
        else:
            report_lines += ["", "✅ Всё OK. Можно обучать!"]

        report_text = "\n".join(report_lines)
        for line in report_lines:
            self.log(line)
        self.show_info("Анти-баг проверка", report_text)

    def quick_start_training(self):
        self.profile_var.set("Авто (рекомендуется)")
        self.apply_training_profile()
        self.check_readiness(show_popup=True)
        if self.count_dataset_images() < 20:
            self.show_warning(self.msg("error_title"), self.msg("low_data"))
            return
        self.start_training()

    def full_auto_mode(self):
        self.run_async(self._full_auto_flow)

    def progressive_auto_mode(self):
        self.run_async(self._progressive_auto_flow)

    def progressive_custom_mode(self):
        self.run_async(self._progressive_custom_flow)

    def _full_auto_flow(self):
        self.set_status(self.msg("status_auto_start"))
        self.profile_var.set("Авто (рекомендуется)")
        self.apply_training_profile()

        image_count = self.count_dataset_images()
        target_min = 200
        if image_count >= target_min:
            self.log(f"Auto mode: в датасете уже {image_count} изображений — скачивание пропущено.")
        else:
            need = max(100, target_min - image_count)
            self.count_var.set(need)
            self.log(f"Auto mode: в датасете {image_count} изображений. Пробуем скачать ещё ~{need}...")
            try:
                self.download_images(show_popup=False)
            except Exception as e:
                self.log(f"⚠️ Скачивание не удалось ({e}). Продолжаем с текущим датасетом ({self.count_dataset_images()} изображений).")

        self.check_readiness(show_popup=False)
        if self.count_dataset_images() < 20:
            self.set_status(self.msg("status_auto_stop"))
            raise ValueError("Авто-режим: недостаточно данных для обучения (минимум 20 изображений).")

        self.train_gan(show_popup=False)
        self.generate_images(show_popup=False)
        self.set_status(self.msg("status_auto_done"))
        self.show_info(self.msg("done_title"), self.msg("auto_done"))

    def _progressive_auto_flow(self):
        self.set_status(self.msg("status_auto_start"))
        self.log("Progressive mode started: 64 -> 128 -> 256")

        image_count = self.count_dataset_images()
        target_min = 300
        if image_count >= target_min:
            self.log(f"Progressive mode: в датасете уже {image_count} изображений — скачивание пропущено.")
        else:
            need = max(150, target_min - image_count)
            self.count_var.set(need)
            self.log(f"Progressive mode: в датасете {image_count} изображений. Пробуем скачать ещё ~{need}...")
            try:
                self.download_images(show_popup=False)
            except Exception as e:
                self.log(f"⚠️ Скачивание не удалось ({e}). Продолжаем с текущим датасетом ({self.count_dataset_images()} изображений).")

        self.check_readiness(show_popup=False)
        if self.count_dataset_images() < 20:
            self.set_status(self.msg("status_auto_stop"))
            raise ValueError("Прогрессивный режим: недостаточно данных (минимум 20 изображений).")

        # Keep user-configured epochs and adapt per stage.
        base_epochs = int(self.epochs_var.get())
        base_batch = int(self.batch_var.get())

        # Stage 1: 64 from scratch.
        self.size_var.set(64)
        self.batch_var.set(max(2, min(base_batch, 16)))
        self.epochs_var.set(max(8, int(base_epochs * 0.5)))
        self.log(
            f"Stage 1/3: train 64px | epochs={self.epochs_var.get()} | batch={self.batch_var.get()} (from scratch)"
        )
        self.train_gan(show_popup=False, resume_checkpoint="")
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Stage 1 failed: checkpoint not created.")

        # Stage 2: 128 fine-tune from 64.
        self.resume_ckpt_var.set(ckpt)
        self.size_var.set(128)
        self.batch_var.set(max(1, min(base_batch, 8)))
        self.epochs_var.set(max(8, int(base_epochs * 0.35)))
        self.log(
            f"Stage 2/3: fine-tune 128px | epochs={self.epochs_var.get()} | batch={self.batch_var.get()} | ckpt={ckpt}"
        )
        self.train_gan(show_popup=False, resume_checkpoint=ckpt, allow_resolution_upgrade=True)
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Stage 2 failed: checkpoint not created.")

        # Stage 3: 256 fine-tune only if CUDA available, else finish at 128.
        if "CUDA" in self.device_name:
            self.resume_ckpt_var.set(ckpt)
            self.size_var.set(256)
            self.batch_var.set(max(1, min(base_batch, 4)))
            self.epochs_var.set(max(6, int(base_epochs * 0.25)))
            self.log(
                f"Stage 3/3: fine-tune 256px | epochs={self.epochs_var.get()} | batch={self.batch_var.get()} | ckpt={ckpt}"
            )
            self.train_gan(show_popup=False, resume_checkpoint=ckpt, allow_resolution_upgrade=True)
            ckpt = self.newest_checkpoint_path()
            if not ckpt:
                raise ValueError("Stage 3 failed: checkpoint not created.")
        else:
            self.log("Stage 3/3 skipped: 256px requires CUDA for practical training speed.")

        self.ckpt_var.set(ckpt)
        self.generate_images(show_popup=False)
        self.set_status(self.msg("status_auto_done"))
        self.show_info(self.msg("done_title"), f"Прогрессивное обучение завершено.\nПоследний чекпоинт:\n{ckpt}")

    def _progressive_custom_flow(self):
        self.set_status(self.msg("status_auto_start"))
        self.log("Progressive custom mode started.")

        ep64 = max(1, int(self.prog_ep64_var.get()))
        bs64 = max(1, int(self.prog_bs64_var.get()))
        ep128 = max(1, int(self.prog_ep128_var.get()))
        bs128 = max(1, int(self.prog_bs128_var.get()))
        ep256 = max(1, int(self.prog_ep256_var.get()))
        bs256 = max(1, int(self.prog_bs256_var.get()))

        image_count = self.count_dataset_images()
        if image_count < 20:
            self.set_status(self.msg("status_auto_stop"))
            raise ValueError("Кастом прогрессия: недостаточно данных (минимум 20 изображений).")

        # 64 stage
        self.size_var.set(64)
        self.epochs_var.set(ep64)
        self.batch_var.set(bs64)
        self.log(f"Custom 1/3: 64px | epochs={ep64} | batch={bs64}")
        self.train_gan(show_popup=False, resume_checkpoint="")
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Custom stage 64 failed.")

        # 128 stage
        self.resume_ckpt_var.set(ckpt)
        self.size_var.set(128)
        self.epochs_var.set(ep128)
        self.batch_var.set(bs128)
        self.log(f"Custom 2/3: 128px | epochs={ep128} | batch={bs128} | ckpt={ckpt}")
        self.train_gan(show_popup=False, resume_checkpoint=ckpt, allow_resolution_upgrade=True)
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Custom stage 128 failed.")

        # 256 stage
        if "CUDA" in self.device_name:
            self.resume_ckpt_var.set(ckpt)
            self.size_var.set(256)
            self.epochs_var.set(ep256)
            self.batch_var.set(bs256)
            self.log(f"Custom 3/3: 256px | epochs={ep256} | batch={bs256} | ckpt={ckpt}")
            self.train_gan(show_popup=False, resume_checkpoint=ckpt, allow_resolution_upgrade=True)
            ckpt = self.newest_checkpoint_path()
            if not ckpt:
                raise ValueError("Custom stage 256 failed.")
        else:
            self.log("Custom stage 256 skipped: CUDA required for practical speed.")

        self.ckpt_var.set(ckpt)
        self.generate_images(show_popup=False)
        self.set_status(self.msg("status_auto_done"))
        self.show_info(self.msg("done_title"), f"Кастом прогрессия завершена.\nПоследний чекпоинт:\n{ckpt}")

    def _build_generate_tab(self):
        # Scrollable canvas — как в вкладке Обучение
        canvas = tk.Canvas(self.tab_generate, bg="#141821", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_generate, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas, padding=10)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        def _mw(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<MouseWheel>", _mw)
        sf.after(100, lambda: self._bind_mousewheel(sf, _mw))

        row = 0
        ttk.Label(sf, text="Шаг 3: Нажмите Сгенерировать. Если поле чекпоинта пустое — берётся последний.",
                  style="Hint.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8)); row+=1

        ttk.Label(sf, text="Чекпоинт (.pt):").grid(row=row, column=0, sticky="w", pady=4)
        self.ckpt_var = tk.StringVar(value="")
        ttk.Entry(sf, textvariable=self.ckpt_var, width=55).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(sf, text="Последний", command=lambda: self.ckpt_var.set(self.newest_checkpoint_path())).grid(
            row=row, column=2, sticky="w", padx=4, pady=4); row+=1

        ttk.Label(sf, text="Сколько картинок:").grid(row=row, column=0, sticky="w", pady=4)
        self.gen_count_var = tk.IntVar(value=1)
        ttk.Spinbox(sf, from_=1, to=64, textvariable=self.gen_count_var, width=12).grid(
            row=row, column=1, sticky="w", padx=6, pady=4); row+=1

        ttk.Label(sf, text="Truncation:").grid(row=row, column=0, sticky="w", pady=4)
        self.truncation_var = tk.DoubleVar(value=1.0)
        trunc_frame = ttk.Frame(sf)
        trunc_frame.grid(row=row, column=1, sticky="w", padx=6, pady=4)
        ttk.Scale(trunc_frame, from_=0.1, to=1.0, variable=self.truncation_var,
                  orient="horizontal", length=180).pack(side="left")
        self.trunc_label = ttk.Label(trunc_frame, text="1.00", width=5)
        self.trunc_label.pack(side="left", padx=4)
        self.truncation_var.trace_add("write", lambda *_: self.trunc_label.configure(text=f"{self.truncation_var.get():.2f}"))
        ttk.Label(sf, text="(0.5=качество, 1.0=разнообразие)", style="Hint.TLabel").grid(
            row=row, column=2, sticky="w", padx=4); row+=1

        ttk.Label(sf, text="Seed (−1=случайный):").grid(row=row, column=0, sticky="w", pady=4)
        self.seed_var = tk.IntVar(value=-1)
        ttk.Spinbox(sf, from_=-1, to=999999, textvariable=self.seed_var, width=12).grid(
            row=row, column=1, sticky="w", padx=6, pady=4); row+=1

        # Real-ESRGAN блок
        esrgan_frame = ttk.LabelFrame(sf, text="Real-ESRGAN — апскейл", padding=6)
        esrgan_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=0, pady=(8, 4)); row+=1
        self.esrgan_mode_var = tk.StringVar(value="Рисунок")
        ttk.Label(esrgan_frame, text="Режим:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(esrgan_frame, text="🖼 Рисунок/Аниме", variable=self.esrgan_mode_var,
                        value="Рисунок", command=self._refresh_model_list).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Radiobutton(esrgan_frame, text="📷 Фото/Реализм", variable=self.esrgan_mode_var,
                        value="Фото", command=self._refresh_model_list).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(esrgan_frame, text="Модель:").grid(row=1, column=0, sticky="w", pady=4)
        self.esrgan_model_var = tk.StringVar(value="Авто (встроенная)")
        self.esrgan_model_combo = ttk.Combobox(esrgan_frame, textvariable=self.esrgan_model_var,
                                                state="readonly", width=40)
        self.esrgan_model_combo.grid(row=1, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Button(esrgan_frame, text="🔄", width=3, command=self._refresh_model_list).grid(row=1, column=3, padx=4)
        ttk.Label(esrgan_frame, text=f"Папка: models/  |  .pth .pt .safetensors .ckpt .bin",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=4, sticky="w")

        ttk.Label(sf, text="Апскейл:").grid(row=row, column=0, sticky="w", pady=4)
        self.upscale_var = tk.StringVar(value="Нет")
        ttk.Combobox(sf, values=["Нет", "x2 (Lanczos)", "x4 (Lanczos)", "x2 (Real-ESRGAN)", "x4 (Real-ESRGAN)"],
                     textvariable=self.upscale_var, state="readonly", width=24).grid(
            row=row, column=1, sticky="w", padx=6, pady=4); row+=1

        btn_frame = ttk.Frame(sf)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=8); row+=1
        ttk.Button(btn_frame, text="Сгенерировать", command=self.start_generation,
                   style="Primary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Интерполяция (latent)",
                   command=self.start_interpolation).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Обновить предпросмотр",
                   command=self.refresh_generated_preview).pack(side="left")

        ds_frame = ttk.LabelFrame(sf, text="Инструменты датасета", padding=8)
        ds_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4)); row+=1
        ttk.Button(ds_frame, text="📊 Анализ", command=self.start_analyze_dataset).pack(side="left", padx=4)
        ttk.Button(ds_frame, text="✂️ Очистка по соотношению", command=self.start_aspect_cleanup).pack(side="left", padx=4)
        ttk.Button(ds_frame, text="🔁 Аугментировать", command=self.start_augment_dataset).pack(side="left", padx=4)

        ttk.Label(sf, text="Предпросмотр:").grid(row=row, column=0, sticky="nw", pady=(8, 4))
        self.gen_preview_label = ttk.Label(sf, text="Нет картинки")
        self.gen_preview_label.grid(row=row, column=1, columnspan=2, sticky="w", padx=6, pady=(8, 4)); row+=1

        # Сравнение двух чекпоинтов
        cmp_frame = ttk.LabelFrame(sf, text="Сравнение двух чекпоинтов", padding=6)
        cmp_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 4)); row+=1

        ttk.Label(cmp_frame, text="Чекпоинт A:").grid(row=0, column=0, sticky="w", pady=2)
        self.cmp_ckpt_a_var = tk.StringVar(value="")
        ttk.Entry(cmp_frame, textvariable=self.cmp_ckpt_a_var, width=44).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(cmp_frame, text="BEST",
                   command=lambda: self.cmp_ckpt_a_var.set(self._newest_best_checkpoint())).grid(row=0, column=2, padx=2)

        ttk.Label(cmp_frame, text="Чекпоинт B:").grid(row=1, column=0, sticky="w", pady=2)
        self.cmp_ckpt_b_var = tk.StringVar(value="")
        ttk.Entry(cmp_frame, textvariable=self.cmp_ckpt_b_var, width=44).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(cmp_frame, text="Последний",
                   command=lambda: self.cmp_ckpt_b_var.set(self.newest_checkpoint_path())).grid(row=1, column=2, padx=2)

        ttk.Button(cmp_frame, text="🔍 Сравнить (левая=A, правая=B)",
                   command=self.start_compare_checkpoints,
                   style="Primary.TButton").grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 2))

        ttk.Label(cmp_frame, text="Результат:").grid(row=3, column=0, sticky="nw", pady=4)
        self.cmp_preview_label = ttk.Label(cmp_frame, text="Нет результата")
        self.cmp_preview_label.grid(row=3, column=1, columnspan=2, sticky="w", padx=4)
        cmp_frame.columnconfigure(1, weight=1)

        sf.columnconfigure(1, weight=1)
        self.root.after(200, self._refresh_model_list)

    def _bind_mousewheel(self, widget, callback):
        widget.bind("<MouseWheel>", callback)
        for child in widget.winfo_children():
            self._bind_mousewheel(child, callback)

    def _build_video_tab(self):
        # Scrollable
        canvas = tk.Canvas(self.tab_video, bg="#141821", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_video, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas, padding=10)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        def _mw(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<MouseWheel>", _mw)
        sf.after(100, lambda: self._bind_mousewheel(sf, _mw))

        ttk.Label(sf, text="Temporal GAN: LSTM + DCGAN генерирует связные последовательности кадров.",
                  style="Hint.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # Переключатель датасета
        ds_frame = ttk.LabelFrame(sf, text="Датасет для обучения", padding=6)
        ds_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        self.vid_dataset_var = tk.StringVar(value="images")
        ttk.Radiobutton(ds_frame, text=f"🖼 Картинки  (dataset/)",
                        variable=self.vid_dataset_var, value="images",
                        command=self._on_vid_dataset_change).pack(side="left", padx=8)
        ttk.Radiobutton(ds_frame, text=f"🎬 Видео  (dataset_video/)",
                        variable=self.vid_dataset_var, value="video",
                        command=self._on_vid_dataset_change).pack(side="left", padx=8)
        self.vid_dataset_hint = ttk.Label(ds_frame,
                                           text="Видео из dataset/ конвертируются в кадры автоматически",
                                           style="Hint.TLabel")
        self.vid_dataset_hint.pack(side="left", padx=8)

        r = 2
        ttk.Label(sf, text="Эпохи:").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_epochs_var = tk.IntVar(value=30)
        ttk.Spinbox(sf, from_=5, to=500, textvariable=self.vid_epochs_var, width=10).grid(
            row=r, column=1, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Кадров в видео:").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_frames_var = tk.IntVar(value=16)
        ttk.Combobox(sf, values=[8, 12, 16, 24, 32], textvariable=self.vid_frames_var,
                     state="readonly", width=8).grid(row=r, column=1, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="FPS:").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_fps_var = tk.IntVar(value=8)
        ttk.Spinbox(sf, from_=4, to=30, textvariable=self.vid_fps_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Размер кадра:").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_size_var = tk.IntVar(value=64)
        ttk.Combobox(sf, values=[64, 128], textvariable=self.vid_size_var,
                     state="readonly", width=8).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Label(sf, text="(128 требует больше VRAM)", style="Hint.TLabel").grid(
            row=r, column=2, sticky="w", padx=4); r+=1

        ttk.Label(sf, text="Batch size:").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_batch_var = tk.IntVar(value=4)
        ttk.Spinbox(sf, from_=1, to=32, textvariable=self.vid_batch_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Чекпоинт (.pt):").grid(row=r, column=0, sticky="w", pady=4)
        self.vid_ckpt_var = tk.StringVar(value="")
        ttk.Entry(sf, textvariable=self.vid_ckpt_var, width=52).grid(
            row=r, column=1, sticky="ew", padx=6)
        ttk.Button(sf, text="Последний",
                   command=lambda: self.vid_ckpt_var.set(self._newest_video_checkpoint())).grid(
            row=r, column=2, padx=4); r+=1

        btn_frame = ttk.Frame(sf)
        btn_frame.grid(row=r, column=0, columnspan=3, sticky="w", pady=10); r+=1
        ttk.Button(btn_frame, text="▶ Обучить Видео-GAN",
                   command=self.start_video_training, style="Primary.TButton").pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="🎬 Сгенерировать видео",
                   command=self.start_video_generation).pack(side="left", padx=(0, 8))

        ttk.Label(sf,
                  text="ℹ  Видео → output/generated_video.gif + .mp4 (если есть imageio).\n"
                       "   Режим 'Картинки': GAN учится плавному движению между разными изображениями.\n"
                       "   Режим 'Видео': GAN учится на реальных кадрах из dataset_video/.\n"
                       "   Работает на DirectML/AMD/NVIDIA/CPU.",
                  style="Hint.TLabel").grid(row=r, column=0, columnspan=3, sticky="w", pady=(8, 4)); r+=1

        ttk.Label(sf, text="Предпросмотр:").grid(row=r, column=0, sticky="nw", pady=(8, 4))
        self.vid_preview_label = ttk.Label(sf, text="Нет видео")
        self.vid_preview_label.grid(row=r, column=1, columnspan=2, sticky="w", padx=6)
        sf.columnconfigure(1, weight=1)

    def _on_vid_dataset_change(self):
        mode = self.vid_dataset_var.get()
        if mode == "images":
            self.vid_dataset_hint.configure(
                text="Модель учится плавным переходам между картинками")
        else:
            self.vid_dataset_hint.configure(
                text=f"Видеофайлы из dataset_video/ разбиваются на кадры")

    def _newest_video_checkpoint(self):
        ckpts = sorted(
            [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)
             if f.startswith("video_gan_") and f.endswith(".pt")],
            key=os.path.getmtime)
        return ckpts[-1] if ckpts else ""

    def start_video_training(self):
        self.run_async(self._video_training_flow)

    def start_video_generation(self):
        self.run_async(self._video_generation_flow)

    def _video_training_flow(self):
        epochs = int(self.vid_epochs_var.get())
        frames = int(self.vid_frames_var.get())
        size = int(self.vid_size_var.get())
        batch = int(self.vid_batch_var.get())
        dataset_mode = self.vid_dataset_var.get()
        self.log(f"=== ВИДЕО-GAN: epochs={epochs}, frames={frames}, size={size}, batch={batch}, "
                 f"датасет={'видео' if dataset_mode=='video' else 'картинки'} ===")

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if dataset_mode == "video":
            # Режим видео: извлекаем кадры из файлов в dataset_video/
            image_files = self._extract_video_frames(size)
            if len(image_files) < frames:
                raise ValueError(f"Мало кадров ({len(image_files)}). Нужно минимум {frames}. "
                                 f"Добавь видеофайлы в dataset_video/")
            self.log(f"Извлечено {len(image_files)} кадров из видео датасета")
        else:
            # Режим картинок
            image_files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                           if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
            if len(image_files) < 20:
                raise ValueError("Нужно минимум 20 изображений в dataset/")
            self.log(f"Используется {len(image_files)} картинок из dataset/")

        dataset = FlatImageDataset(image_files, transform=transform)
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

        netVG = VideoGenerator(latent_dim=LATENT_DIM, image_size=size, n_frames=frames).to(self.device)
        netVD = VideoDiscriminator(image_size=size, n_frames=frames).to(self.device)
        netVG.apply(weights_init)
        netVD.apply(weights_init)

        optimG = optim.Adam(netVG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimD = optim.Adam(netVD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(1, LATENT_DIM, device=self.device)

        for epoch in range(epochs):
            for step, (real_imgs, _) in enumerate(loader):
                b = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                # Строим real_video покадрово на CPU, затем передаём в D покадрово
                # Избегаем 5D тензоров на DirectML устройстве полностью
                # real_imgs: [B, C, H, W] — добавляем лёгкий шум для каждого кадра отдельно

                label_real = torch.ones(b, device=self.device) * 0.95
                label_fake = torch.zeros(b, device=self.device) + 0.05

                # ── Обучаем D ──
                netVD.zero_grad()
                # Передаём кадры по одному (frame list) вместо 5D тензора
                real_frame_list = []
                for t in range(frames):
                    noise_t = torch.randn_like(real_imgs) * 0.05
                    real_frame_list.append(real_imgs + noise_t)

                noise_g = torch.randn(b, LATENT_DIM, device=self.device)
                fake_frame_list = netVG(noise_g)  # список кадров [T, B, C, H, W]

                out_real = netVD(real_frame_list)
                loss_d_real = criterion(out_real, label_real)

                out_fake = netVD(fake_frame_list)
                loss_d_fake = criterion(out_fake.detach(), label_fake)
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(netVD.parameters(), 1.0)
                optimD.step()

                # ── Обучаем G ──
                netVG.zero_grad()
                noise_g2 = torch.randn(b, LATENT_DIM, device=self.device)
                fake_frame_list2 = netVG(noise_g2)
                out_gen = netVD(fake_frame_list2)
                loss_g = criterion(out_gen, torch.ones(b, device=self.device) * 0.95)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(netVG.parameters(), 1.0)
                optimG.step()

                if (step + 1) % 10 == 0:
                    self.log(f"Video Epoch {epoch+1}/{epochs} | Step {step+1}/{len(loader)} | "
                             f"Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

            with torch.no_grad():
                # netVG возвращает список кадров [B, C, H, W] × n_frames
                preview_frames = netVG(fixed_noise)  # список [1, C, H, W]
                # Склеиваем кадры в одну строку для превью
                preview_grid = torch.cat([f.detach().cpu() for f in preview_frames], dim=0)
                preview_path = os.path.join(OUTPUT_DIR, f"video_preview_epoch_{epoch+1}.jpg")
                utils.save_image(preview_grid, preview_path, normalize=True, nrow=min(8, frames))
                self._ui_call(self._update_image_preview_widget, self.vid_preview_label, preview_path)

        ckpt_name = f"video_gan_{size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        torch.save({"generator": netVG.state_dict(), "discriminator": netVD.state_dict(),
                    "image_size": size, "n_frames": frames}, ckpt_path)
        self.vid_ckpt_var.set(ckpt_path)
        self.log(f"✓ Видео-GAN обучен. Чекпоинт: {ckpt_name}")
        self.show_info("Готово", f"Видео-GAN обучен!\nЧекпоинт: {ckpt_name}")

    def _extract_video_frames(self, size: int) -> list:
        """Извлекает кадры из видеофайлов dataset_video/ во временную папку и возвращает список путей."""
        try:
            import cv2
        except ImportError:
            self.log("Устанавливаю opencv-python для работы с видео...")
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
            import cv2

        frames_dir = os.path.join(OUTPUT_DIR, "_video_frames_tmp")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = []
        vid_files = [os.path.join(DATASET_VIDEO_DIR, f) for f in os.listdir(DATASET_VIDEO_DIR)
                     if f.lower().endswith(ALLOWED_VIDEO_EXTENSIONS)]
        if not vid_files:
            raise ValueError(f"В dataset_video/ нет видеофайлов. Добавь .mp4/.avi/.mov/.gif и т.д.")

        for vid_path in vid_files:
            cap = cv2.VideoCapture(vid_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // 30)  # берём до 30 кадров из каждого видео
            idx = 0
            saved = 0
            while cap.isOpened() and saved < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % step == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    fname = f"frame_{os.path.basename(vid_path)}_{idx:05d}.jpg"
                    img.save(os.path.join(frames_dir, fname), quality=90)
                    frame_paths.append(os.path.join(frames_dir, fname))
                    saved += 1
                idx += 1
            cap.release()
        return frame_paths

    def _video_generation_flow(self):
        ckpt = self.vid_ckpt_var.get().strip() or self._newest_video_checkpoint()
        if not ckpt or not os.path.isfile(ckpt):
            raise ValueError("Чекпоинт Видео-GAN не найден. Сначала обучите модель.")

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        size = int(state.get("image_size", 64))
        frames = int(state.get("n_frames", 16))
        fps = int(self.vid_fps_var.get())

        netVG = VideoGenerator(latent_dim=LATENT_DIM, image_size=size, n_frames=frames).to(self.device)
        netVG.load_state_dict(state["generator"], strict=False)
        netVG.eval()

        seed = int(self.seed_var.get()) if hasattr(self, "seed_var") else -1
        if seed >= 0:
            torch.manual_seed(seed)

        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM, device=self.device)
            frame_list = netVG(noise)  # список [1, C, H, W] × n_frames

        # Конвертируем список кадров в PIL
        pil_frames = []
        for frame_tensor in frame_list:
            frame = frame_tensor[0].detach().cpu()   # [C, H, W]
            frame = (frame * 0.5 + 0.5).clamp(0, 1)
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil_frames.append(Image.fromarray(frame_np))

        # Сохраняем как GIF
        gif_path = os.path.join(OUTPUT_DIR, "generated_video.gif")
        duration_ms = int(1000 / fps)
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=duration_ms)
        self.log(f"✓ Видео сохранено: {gif_path} ({frames} кадров, {fps} FPS)")

        # Пробуем сохранить MP4 через imageio если есть
        try:
            import imageio
            mp4_path = os.path.join(OUTPUT_DIR, "generated_video.mp4")
            writer = imageio.get_writer(mp4_path, fps=fps)
            for frame in pil_frames:
                import numpy as np
                writer.append_data(np.array(frame))
            writer.close()
            self.log(f"✓ MP4 сохранён: {mp4_path}")
        except Exception:
            self.log("ℹ MP4 не создан (нет imageio). GIF доступен.")

        # Показываем первый кадр как предпросмотр
        self._ui_call(self._update_image_preview_widget, self.vid_preview_label, gif_path)
        self.show_info("Готово", f"Видео сгенерировано!\nGIF: {gif_path}")

    def _build_gallery_tab(self):
        top = ttk.Frame(self.tab_gallery)
        top.pack(fill="x", pady=(0, 6))
        ttk.Button(top, text="🔄 Обновить", command=self._refresh_gallery).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="🗑 Удалить выбранные", command=self._delete_selected_gallery).pack(side="left", padx=(0, 8))
        ttk.Button(top, text="📂 Открыть папку output/",
                   command=lambda: os.startfile(OUTPUT_DIR) if os.name == "nt" else None).pack(side="left")

        # Canvas + scrollbar для сетки картинок
        canvas = tk.Canvas(self.tab_gallery, bg="#0F1320", highlightthickness=0)
        sb = ttk.Scrollbar(self.tab_gallery, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        self._gallery_canvas = canvas
        self._gallery_vars = {}  # path -> BooleanVar (выбрана ли)
        self._gallery_images = []  # держим ссылки на PhotoImage

        self.tab_gallery.after(300, self._refresh_gallery)

    def _refresh_gallery(self):
        canvas = self._gallery_canvas
        canvas.delete("all")
        self._gallery_images.clear()
        self._gallery_vars.clear()

        files = sorted([
            os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "generated" in f
        ], key=os.path.getmtime, reverse=True)

        if not files:
            canvas.create_text(300, 100, text="Нет сгенерированных картинок в output/",
                               fill="#95A2BD", font=("Segoe UI", 11))
            return

        thumb_size = 120
        cols = 6
        pad = 8
        for i, path in enumerate(files):
            col = i % cols
            row = i // cols
            x = pad + col * (thumb_size + pad)
            y = pad + row * (thumb_size + pad + 20)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((thumb_size, thumb_size))
                tk_img = ImageTk.PhotoImage(img)
                self._gallery_images.append(tk_img)
                canvas.create_image(x, y, anchor="nw", image=tk_img)
            except Exception:
                canvas.create_rectangle(x, y, x+thumb_size, y+thumb_size, fill="#1A2035")

            # Чекбокс под картинкой
            var = tk.BooleanVar(value=False)
            self._gallery_vars[path] = var
            cb = ttk.Checkbutton(canvas, variable=var)
            canvas.create_window(x + thumb_size // 2, y + thumb_size + 10, window=cb)

        total_rows = (len(files) + cols - 1) // cols
        canvas.configure(scrollregion=(0, 0, cols*(thumb_size+pad)+pad,
                                       total_rows*(thumb_size+pad+20)+pad))

    def _delete_selected_gallery(self):
        to_delete = [p for p, v in self._gallery_vars.items() if v.get()]
        if not to_delete:
            self.show_info("Галерея", "Ничего не выбрано.")
            return
        for path in to_delete:
            try:
                os.remove(path)
            except Exception:
                pass
        self.log(f"Галерея: удалено {len(to_delete)} файлов")
        self._refresh_gallery()

    def _build_history_tab(self):
        ttk.Label(self.tab_history,
                  text="История запусков обучения. Каждый запуск сохраняется автоматически.",
                  style="Hint.TLabel").pack(anchor="w", pady=(0, 6))

        btn_f = ttk.Frame(self.tab_history)
        btn_f.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_f, text="🔄 Обновить", command=self._refresh_history).pack(side="left", padx=(0, 8))
        ttk.Button(btn_f, text="🗑 Очистить историю", command=self._clear_history).pack(side="left")

        cols = ("Дата", "Размер", "Эпохи", "Batch", "Финал Loss D", "Финал Loss G", "Чекпоинт")
        self._history_tree = ttk.Treeview(self.tab_history, columns=cols, show="headings", height=12)
        for c in cols:
            self._history_tree.heading(c, text=c)
            self._history_tree.column(c, width=110, anchor="center")
        self._history_tree.pack(fill="both", expand=True)

        sb = ttk.Scrollbar(self.tab_history, orient="vertical", command=self._history_tree.yview)
        self._history_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # График для выбранного запуска
        self._history_plot = tk.Canvas(self.tab_history, height=160, bg="#0F1320",
                                       highlightthickness=1, highlightbackground="#2B3550")
        self._history_plot.pack(fill="x", pady=(8, 0))
        self._history_tree.bind("<<TreeviewSelect>>", self._on_history_select)
        self.tab_history.after(400, self._refresh_history)

    def _refresh_history(self):
        self._history_tree.delete(*self._history_tree.get_children())
        history_file = os.path.join(APP_DIR, "training_history.jsonl")
        if not os.path.isfile(history_file):
            return
        with open(history_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    self._history_tree.insert("", "end", values=(
                        rec.get("date", "?"),
                        rec.get("size", "?"),
                        rec.get("epochs", "?"),
                        rec.get("batch", "?"),
                        f"{rec.get('final_loss_d', 0):.4f}",
                        f"{rec.get('final_loss_g', 0):.4f}",
                        os.path.basename(rec.get("checkpoint", "")),
                    ))
                except Exception:
                    pass

    def _on_history_select(self, _event):
        pass  # Можно добавить отображение графика конкретного запуска

    def _clear_history(self):
        history_file = os.path.join(APP_DIR, "training_history.jsonl")
        if os.path.isfile(history_file):
            os.remove(history_file)
        self._refresh_history()
        self.log("История обучений очищена.")

    def _save_training_record(self, **kwargs):
        """Записывает один запуск в training_history.jsonl."""
        history_file = os.path.join(APP_DIR, "training_history.jsonl")
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _build_settings_tab(self):
        canvas = tk.Canvas(self.tab_settings, bg="#141821", highlightthickness=0)
        sb = ttk.Scrollbar(self.tab_settings, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas, padding=10)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        def _mw(e): canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind("<MouseWheel>", _mw)

        r = 0
        # ── Темп / Интерфейс ──
        ttk.Label(sf, text="Интерфейс", style="Title.TLabel").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r+=1

        ttk.Label(sf, text="Тема:").grid(row=r, column=0, sticky="w", pady=4)
        self.theme_var = tk.StringVar(value="Тёмная")
        ttk.Combobox(sf, values=["Тёмная", "Светлая"], textvariable=self.theme_var,
                     state="readonly", width=14).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Button(sf, text="Применить", command=self._apply_theme).grid(row=r, column=2, padx=4); r+=1

        ttk.Label(sf, text="Язык UI:").grid(row=r, column=0, sticky="w", pady=4)
        self.lang_var = tk.StringVar(value="Русский")
        ttk.Combobox(sf, values=["Русский", "English"], textvariable=self.lang_var,
                     state="readonly", width=14).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Button(sf, text="Применить", command=self.set_language).grid(row=r, column=2, padx=4); r+=1

        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r+=1

        # ── Обучение ──
        ttk.Label(sf, text="Обучение", style="Title.TLabel").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r+=1

        ttk.Label(sf, text="Gradient accumulation:").grid(row=r, column=0, sticky="w", pady=4)
        self.grad_accum_var = tk.IntVar(value=1)
        ttk.Spinbox(sf, from_=1, to=16, textvariable=self.grad_accum_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Label(sf, text="(1=выкл, 4=batch×4 без лишней памяти)", style="Hint.TLabel").grid(
            row=r, column=2, sticky="w", padx=4); r+=1

        ttk.Label(sf, text="LR Warmup эпох:").grid(row=r, column=0, sticky="w", pady=4)
        self.warmup_epochs_var = tk.IntVar(value=3)
        ttk.Spinbox(sf, from_=0, to=20, textvariable=self.warmup_epochs_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Label(sf, text="(плавный старт LR с нуля)", style="Hint.TLabel").grid(
            row=r, column=2, sticky="w", padx=4); r+=1

        ttk.Label(sf, text="FID оценка каждые N эпох:").grid(row=r, column=0, sticky="w", pady=4)
        self.fid_every_var = tk.IntVar(value=0)
        ttk.Spinbox(sf, from_=0, to=50, textvariable=self.fid_every_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Label(sf, text="(0=выкл, медленно но полезно)", style="Hint.TLabel").grid(
            row=r, column=2, sticky="w", padx=4); r+=1

        ttk.Label(sf, text="Early stopping (эпох без улучшения):").grid(row=r, column=0, sticky="w", pady=4)
        self.early_stop_var = tk.IntVar(value=0)
        ttk.Spinbox(sf, from_=0, to=50, textvariable=self.early_stop_var, width=8).grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Label(sf, text="(0=выкл)", style="Hint.TLabel").grid(
            row=r, column=2, sticky="w", padx=4); r+=1

        ttk.Label(sf, text="Mixed Precision (FP16):").grid(row=r, column=0, sticky="w", pady=4)
        self.fp16_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, variable=self.fp16_var,
                        text="Включить (CUDA быстрее, DirectML — нестабильно)").grid(
            row=r, column=1, columnspan=2, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Multi-GPU (CUDA):").grid(row=r, column=0, sticky="w", pady=4)
        self.multi_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, variable=self.multi_gpu_var,
                        text="DataParallel (только CUDA, игнорируется на DirectML)").grid(
            row=r, column=1, columnspan=2, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Звук по завершению:").grid(row=r, column=0, sticky="w", pady=4)
        self.sound_notify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, variable=self.sound_notify_var, text="Включить (Windows)").grid(
            row=r, column=1, sticky="w", padx=6); r+=1

        ttk.Label(sf, text="Лог в файл:").grid(row=r, column=0, sticky="w", pady=4)
        self.file_log_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sf, variable=self.file_log_var,
                        text=f"Писать в training_log.txt").grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Button(sf, text="Открыть лог",
                   command=lambda: os.startfile(LOG_FILE) if os.path.isfile(LOG_FILE) and os.name=="nt" else None
                   ).grid(row=r, column=2, padx=4); r+=1

        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r+=1

        # ── Датасет ──
        ttk.Label(sf, text="Инструменты датасета", style="Title.TLabel").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r+=1

        ds_btn_f = ttk.Frame(sf)
        ds_btn_f.grid(row=r, column=0, columnspan=3, sticky="w", pady=4); r+=1
        ttk.Button(ds_btn_f, text="🔁 pHash дедупликация",
                   command=self.start_phash_dedup).pack(side="left", padx=(0, 8))
        ttk.Button(ds_btn_f, text="✂️ Кадрирование лиц (OpenCV)",
                   command=self.start_face_crop).pack(side="left", padx=(0, 8))
        ttk.Button(ds_btn_f, text="⚖️ Анализ баланса датасета",
                   command=self.start_balance_check).pack(side="left")

        ttk.Label(sf, text="CLIP фильтрация:").grid(row=r, column=0, sticky="w", pady=4)
        self.clip_prompt_var = tk.StringVar(value="high quality image")
        ttk.Entry(sf, textvariable=self.clip_prompt_var, width=40).grid(
            row=r, column=1, sticky="w", padx=6)
        ttk.Button(sf, text="▶ Запустить CLIP фильтр",
                   command=self.start_clip_filter).grid(row=r, column=2, padx=4); r+=1
        ttk.Label(sf, text="(скачивает CLIP ~400MB при первом запуске)", style="Hint.TLabel").grid(
            row=r, column=1, columnspan=2, sticky="w", padx=6); r+=1

        ttk.Separator(sf, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", pady=8); r+=1

        # ── Экспорт ──
        ttk.Label(sf, text="Экспорт", style="Title.TLabel").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 6)); r+=1

        exp_f = ttk.Frame(sf)
        exp_f.grid(row=r, column=0, columnspan=3, sticky="w", pady=4); r+=1
        ttk.Button(exp_f, text="💾 Экспорт в ONNX",
                   command=self.start_onnx_export).pack(side="left", padx=(0, 8))
        ttk.Button(exp_f, text="📤 Сохранить профиль настроек",
                   command=self.export_profile).pack(side="left", padx=(0, 8))
        ttk.Button(exp_f, text="📥 Загрузить профиль",
                   command=self.import_profile).pack(side="left")

        sf.columnconfigure(1, weight=1)

    def open_settings(self):
        SettingsWindow(self.root)

    def _apply_theme(self):
        """Переключает тёмную/светлую тему."""
        # Может вызываться как из UI (theme_var), так и при запуске (SETTINGS)
        if hasattr(self, "theme_var"):
            theme_key = "Светлая" if self.theme_var.get() == "Светлая" else "Тёмная"
        else:
            theme_key = "Светлая" if SETTINGS.get("theme") == "light" else "Тёмная"
        style = ttk.Style(self.root)
        if theme_key == "Светлая":
            bg, fg, field = "#F0F2F5", "#1A1A2E", "#FFFFFF"
            hint_fg = "#666"
        else:
            bg, fg, field = "#141821", "#E6EAF2", "#202736"
            hint_fg = "#95A2BD"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Hint.TLabel", background=bg, foreground=hint_fg)
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg)
        style.configure("TEntry", fieldbackground=field, foreground=fg)
        style.configure("TSpinbox", fieldbackground=field, foreground=fg)
        style.configure("TCombobox", fieldbackground=field, foreground=fg)
        if hasattr(self, "log_text"):
            log_bg = "#0F1320" if theme_key == "Тёмная" else "#F8F9FA"
            log_fg = "#D5DEF5" if theme_key == "Тёмная" else "#1A1A2E"
            self.log_text.configure(bg=log_bg, fg=log_fg)
        if hasattr(self, "theme_var"):
            self.log(f"Тема: {theme_key}")

    def start_phash_dedup(self):
        self.run_async(self._phash_dedup_flow)

    def _phash_dedup_flow(self):
        self.log("=== pHash ДЕДУПЛИКАЦИЯ ===")
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        hashes = {}
        duplicates = []
        for path in files:
            try:
                img = Image.open(path).convert("L").resize((16, 16))
                import numpy as np
                arr = list(img.getdata())
                avg = sum(arr) / len(arr)
                phash = "".join("1" if p > avg else "0" for p in arr)
                # Проверяем схожесть (расстояние Хэмминга)
                is_dup = False
                for existing_hash, existing_path in hashes.items():
                    diff = sum(a != b for a, b in zip(phash, existing_hash))
                    if diff < 20:  # порог схожести
                        duplicates.append(path)
                        is_dup = True
                        break
                if not is_dup:
                    hashes[phash] = path
            except Exception:
                continue
        removed = 0
        for path in duplicates:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
        self.log(f"Дедупликация завершена: удалено {removed} похожих из {len(files)}")
        self.show_info("pHash дедупликация",
                       f"Проверено: {len(files)}\nУдалено похожих: {removed}\nОсталось: {len(files)-removed}")

    def start_face_crop(self):
        self.run_async(self._face_crop_flow)

    def _face_crop_flow(self):
        self.log("=== КАДРИРОВАНИЕ ЛИЦ ===")
        try:
            import cv2
        except ImportError:
            self.log("Устанавливаю opencv-python...")
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "opencv-python"], check=True)
            import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        cropped = 0
        for path in files:
            try:
                img_cv = cv2.imread(path)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    pad = int(max(w, h) * 0.3)
                    x1 = max(0, x - pad); y1 = max(0, y - pad)
                    x2 = min(img_cv.shape[1], x + w + pad)
                    y2 = min(img_cv.shape[0], y + h + pad)
                    crop = img_cv[y1:y2, x1:x2]
                    cv2.imwrite(path, crop)
                    cropped += 1
            except Exception:
                continue
        self.log(f"Кадрирование лиц: обработано {cropped} из {len(files)}")
        self.show_info("Кадрирование лиц", f"Лиц обнаружено и кадрировано: {cropped}\nИз всего: {len(files)}")

    def start_balance_check(self):
        self.run_async(self._balance_check_flow)

    def _balance_check_flow(self):
        self.log("=== АНАЛИЗ БАЛАНСА ДАТАСЕТА ===")
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        if not files:
            self.show_info("Баланс", "Датасет пуст.")
            return
        portrait = square = landscape = 0
        for path in files:
            try:
                w, h = Image.open(path).size
                r = w / max(h, 1)
                if r > 1.2: landscape += 1
                elif r < 0.8: portrait += 1
                else: square += 1
            except Exception:
                pass
        total = len(files)
        lines = [
            f"Всего: {total}",
            f"Портрет (высокие): {portrait} ({100*portrait//max(total,1)}%)",
            f"Квадратные: {square} ({100*square//max(total,1)}%)",
            f"Пейзаж (широкие): {landscape} ({100*landscape//max(total,1)}%)",
            "",
        ]
        dominant = max(portrait, square, landscape)
        if dominant / total > 0.8:
            lines.append("⚠ Датасет несбалансирован (>80% одного типа).")
            lines.append("Совет: добавь больше разнообразных картинок или используй аугментацию.")
        else:
            lines.append("✓ Датасет относительно сбалансирован.")
        self.log("\n".join(lines))
        self.show_info("Баланс датасета", "\n".join(lines))

    def start_clip_filter(self):
        self.run_async(self._clip_filter_flow)

    def _clip_filter_flow(self):
        prompt = self.clip_prompt_var.get().strip()
        if not prompt:
            raise ValueError("Введи текстовый запрос для CLIP фильтрации")
        self.log(f"=== CLIP ФИЛЬТРАЦИЯ: '{prompt}' ===")
        try:
            import clip
        except ImportError:
            self.log("Устанавливаю clip...")
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install",
                            "git+https://github.com/openai/CLIP.git"], check=True)
            import clip

        import clip as clip_mod
        clip_model, clip_preprocess = clip_mod.load("ViT-B/32", device="cpu")
        text = clip_mod.tokenize([prompt])
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        scores = []
        for path in files:
            try:
                img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    img_feat = clip_model.encode_image(img)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                score = float((img_feat @ text_features.T).item())
                scores.append((score, path))
            except Exception:
                continue

        scores.sort(reverse=True)
        keep = int(len(scores) * 0.7)  # оставляем топ 70%
        removed = 0
        for i, (score, path) in enumerate(scores):
            if i >= keep:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
        self.log(f"CLIP: удалено {removed} из {len(files)}, осталось {len(files)-removed}")
        self.show_info("CLIP фильтрация",
                       f"Удалено {removed} картинок (нижние 30% по релевантности '{prompt}')\n"
                       f"Осталось: {len(files)-removed}")

    def start_onnx_export(self):
        self.run_async(self._onnx_export_flow)

    def _onnx_export_flow(self):
        self.log("=== ЭКСПОРТ В ONNX ===")
        ckpt = self.newest_checkpoint_path()
        if not ckpt:
            raise ValueError("Нет чекпоинта. Сначала обучите модель.")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        image_size = int(state.get("image_size", 64))
        latent_dim = int(state.get("latent_dim", LATENT_DIM))
        netG = Generator(image_size=image_size, latent_dim=latent_dim)
        gen_state = state.get("generator_ema_state_dict", state["generator_state_dict"])
        netG.load_state_dict(gen_state, strict=False)
        netG.eval()
        dummy = torch.randn(1, latent_dim, 1, 1)
        onnx_path = os.path.join(OUTPUT_DIR, f"generator_{image_size}px.onnx")
        torch.onnx.export(
            netG, dummy, onnx_path,
            input_names=["noise"], output_names=["image"],
            dynamic_axes={"noise": {0: "batch"}, "image": {0: "batch"}},
            opset_version=11,
        )
        self.log(f"✓ Экспортировано в ONNX: {onnx_path}")
        self.show_info("ONNX экспорт", f"Генератор сохранён:\n{onnx_path}\n\nМожно использовать в ONNX Runtime, Unity, Unreal и других.")

    def export_profile(self):
        try:
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                defaultextension=".json", filetypes=[("JSON", "*.json")],
                initialfile="gan_profile.json", initialdir=APP_DIR)
            if not path:
                return
            profile = {
                "epochs": self.epochs_var.get(),
                "batch": self.batch_var.get(),
                "size": self.size_var.get(),
                "ema_decay": self.ema_decay_var.get(),
                "diffaug": self.diffaug_var.get(),
                "grad_accum": self.grad_accum_var.get(),
                "warmup_epochs": self.warmup_epochs_var.get(),
                "fid_every": self.fid_every_var.get(),
                "early_stop": self.early_stop_var.get(),
                "fp16": self.fp16_var.get(),
                "sound_notify": self.sound_notify_var.get(),
                "file_log": self.file_log_var.get(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            self.log(f"Профиль сохранён: {path}")
            self.show_info("Профиль", f"Сохранено:\n{path}")
        except Exception as e:
            self.show_error("Ошибка", str(e))

    def import_profile(self):
        try:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                filetypes=[("JSON", "*.json")], initialdir=APP_DIR)
            if not path:
                return
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            self.epochs_var.set(profile.get("epochs", 50))
            self.batch_var.set(profile.get("batch", 8))
            self.size_var.set(profile.get("size", 64))
            self.ema_decay_var.set(profile.get("ema_decay", 0.995))
            self.diffaug_var.set(profile.get("diffaug", True))
            self.grad_accum_var.set(profile.get("grad_accum", 1))
            self.warmup_epochs_var.set(profile.get("warmup_epochs", 3))
            self.fid_every_var.set(profile.get("fid_every", 0))
            self.early_stop_var.set(profile.get("early_stop", 0))
            self.fp16_var.set(profile.get("fp16", False))
            self.sound_notify_var.set(profile.get("sound_notify", True))
            self.file_log_var.set(profile.get("file_log", True))
            self.log(f"Профиль загружен: {path}")
            self.show_info("Профиль", "Настройки применены.")
        except Exception as e:
            self.show_error("Ошибка загрузки профиля", str(e))

    def _play_notification(self):
        """Системный звук по завершению обучения."""
        try:
            if not self.sound_notify_var.get():
                return
            if os.name == "nt":
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass

    def _write_log_file(self, text: str):
        """Дублирует сообщение в файл."""
        try:
            if self.file_log_var.get():
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n")
        except Exception:
            pass

    def _compute_fid_approx(self, netG, loader, n_samples=256) -> float:
        """Упрощённая метрика разнообразия/качества без Inception.
        Считает среднее попарное расстояние между сгенерированными картинками.
        Чем выше — тем больше разнообразие."""
        netG.eval()
        imgs = []
        with torch.no_grad():
            while len(imgs) < n_samples:
                noise = torch.randn(min(16, n_samples - len(imgs)),
                                    LATENT_DIM, 1, 1, device=self.device)
                batch = netG(noise).detach().cpu()
                for i in range(batch.size(0)):
                    flat = batch[i].view(-1).numpy()
                    imgs.append(flat)
        import numpy as np
        arr = np.array(imgs[:n_samples])
        # Попарная косинусная схожесть — меньше = лучше разнообразие
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        normed = arr / (norms + 1e-8)
        sim_matrix = normed @ normed.T
        np.fill_diagonal(sim_matrix, 0)
        mean_sim = sim_matrix.sum() / (n_samples * (n_samples - 1))
        netG.train()
        return float(mean_sim)

    def _refresh_vgg_model_list(self):
        """Сканирует models_vgg/ и обновляет список VGG моделей."""
        found = ["Авто (VGG16 torchvision ~528MB)"]
        if os.path.isdir(VGG_MODELS_DIR):
            for f in sorted(os.listdir(VGG_MODELS_DIR)):
                if f.lower().endswith(ALLOWED_MODEL_EXTENSIONS):
                    found.append(f)
        if hasattr(self, "vgg_model_combo"):
            self.vgg_model_combo["values"] = found
            if self.vgg_model_var.get() not in found:
                self.vgg_model_var.set(found[0])
        self.log(f"VGG моделей в models_vgg/: {len(found)-1}")

    # ── Сравнение двух чекпоинтов ─────────────────────────────────────────────
    def _newest_best_checkpoint(self) -> str:
        ckpts = sorted(
            [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR)
             if f.endswith(".pt") and "BEST" in f],
            key=os.path.getmtime)
        return ckpts[-1] if ckpts else self.newest_checkpoint_path()

    def start_compare_checkpoints(self):
        self.run_async(self._compare_checkpoints_flow)

    def _compare_checkpoints_flow(self):
        ckpt_a = self.cmp_ckpt_a_var.get().strip()
        ckpt_b = self.cmp_ckpt_b_var.get().strip()
        if not ckpt_a or not ckpt_b:
            raise ValueError("Укажи оба чекпоинта (A и B) для сравнения.")
        if not os.path.isfile(ckpt_a):
            raise ValueError(f"Файл A не найден: {ckpt_a}")
        if not os.path.isfile(ckpt_b):
            raise ValueError(f"Файл B не найден: {ckpt_b}")

        self.log(f"Сравнение: A={os.path.basename(ckpt_a)}  B={os.path.basename(ckpt_b)}")

        def load_gen(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sz = int(state.get("image_size", 64))
            ld = int(state.get("latent_dim", LATENT_DIM))
            g = Generator(image_size=sz, latent_dim=ld)
            gen_state = state.get("generator_ema_state_dict", state.get("generator_state_dict", {}))
            g.load_state_dict(gen_state, strict=False)
            g.eval()
            return g, sz, ld

        gA, szA, ldA = load_gen(ckpt_a)
        gB, szB, ldB = load_gen(ckpt_b)
        n = 8
        trunc = float(self.truncation_var.get()) if hasattr(self, "truncation_var") else 1.0

        with torch.no_grad():
            noise_a = torch.randn(n, ldA, 1, 1) * trunc
            noise_b = torch.randn(n, ldB, 1, 1) * trunc
            imgs_a = gA(noise_a).detach().cpu()  # [8, 3, H, W]
            imgs_b = gB(noise_b).detach().cpu()

        # Склеиваем попарно: левая половина A, правая половина B
        from torchvision.transforms.functional import resize
        import torchvision.transforms.functional as TF

        h = max(szA, szB)
        result_frames = []
        for i in range(n):
            fa = TF.resize(imgs_a[i], [h, h]) if szA != h else imgs_a[i]
            fb = TF.resize(imgs_b[i], [h, h]) if szB != h else imgs_b[i]
            left  = fa[:, :, :h//2]
            right = fb[:, :, h//2:]
            merged = torch.cat([left, right], dim=2)  # [3, h, h]
            result_frames.append(merged)

        grid = torch.stack(result_frames, dim=0)
        out_path = os.path.join(OUTPUT_DIR, "comparison_AB.jpg")
        utils.save_image(grid, out_path, normalize=True, nrow=4)
        self.log(f"✓ Сравнение сохранено: {out_path}  (левая=A, правая=B, разделитель по центру)")
        self._ui_call(self._update_image_preview_widget, self.cmp_preview_label, out_path, (480, 480))
        self.show_info("Сравнение готово",
                       f"Левая половина — {os.path.basename(ckpt_a)}\n"
                       f"Правая половина — {os.path.basename(ckpt_b)}\n"
                       f"Сохранено: {out_path}")

    # ── Drag & Drop для датасета ──────────────────────────────────────────────
    def _setup_drag_drop(self):
        """Включает drag & drop на главное окно — перетащи папку или файлы."""
        try:
            # Пробуем tkinterdnd2 (нужно установить отдельно)
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._on_drop)
            self.log("Drag & Drop: включён (tkinterdnd2)")
        except Exception:
            self.log("Drag & Drop: недоступен. Установи: pip install tkinterdnd2")

    def _on_drop(self, event):
        """Обрабатывает перетаскивание файлов/папок в окно."""
        raw = event.data.strip()
        # tkinterdnd2 возвращает пути в фигурных скобках если есть пробелы
        paths = []
        import re as _re
        for p in _re.findall(r'\{[^}]+\}|[^\s]+', raw):
            paths.append(p.strip("{}"))

        img_count = 0
        for path in paths:
            if os.path.isdir(path):
                # Папка — копируем все изображения
                for fname in os.listdir(path):
                    if fname.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                        src_path = os.path.join(path, fname)
                        dst_path = os.path.join(DATASET_DIR, fname)
                        if not os.path.isfile(dst_path):
                            import shutil
                            shutil.copy2(src_path, dst_path)
                            img_count += 1
            elif os.path.isfile(path):
                if path.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                    import shutil
                    dst = os.path.join(DATASET_DIR, os.path.basename(path))
                    if not os.path.isfile(dst):
                        shutil.copy2(path, dst)
                        img_count += 1

        if img_count > 0:
            self.log(f"✓ Drag & Drop: добавлено {img_count} изображений в dataset/")
            self._ui_call(self._refresh_dataset_info)
            self.show_info("Drag & Drop", f"Добавлено {img_count} изображений в dataset/")
        else:
            self.log("Drag & Drop: подходящих изображений не найдено")

    # ── Горячие клавиши ───────────────────────────────────────────────────────
    def _setup_hotkeys(self):
        """Регистрирует горячие клавиши."""
        self.root.bind("<F5>",      lambda e: self.quick_start_training())
        self.root.bind("<Escape>",  lambda e: self._stop_training_hotkey())
        self.root.bind("<space>",   lambda e: self._generate_hotkey(e))
        self.root.bind("<F1>",      lambda e: self._show_hotkeys_help())
        self.root.bind("<F12>",     lambda e: self.run_diagnostics())
        self.log("Горячие клавиши: F5=Обучить  Escape=Стоп  Space=Генерировать  F1=Помощь  F12=Диагностика")

    def _stop_training_hotkey(self):
        if self.busy:
            self.log("⚠ Запрос остановки через Escape... (дождись конца эпохи)")
            self._stop_requested = True

    def _generate_hotkey(self, event):
        # Space работает только если фокус не на поле ввода
        if not isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)):
            self.start_generation()

    def _show_hotkeys_help(self):
        self.show_info("Горячие клавиши",
                       "F5        — Быстрый старт обучения\n"
                       "Escape    — Запрос остановки обучения\n"
                       "Space     — Сгенерировать изображение\n"
                       "F12       — Анти-баг диагностика\n"
                       "F1        — Эта подсказка")

    # ── Автообновление с GitHub ───────────────────────────────────────────────
    def check_for_updates(self):
        self.run_async(self._check_updates_flow)

    def _check_updates_flow(self):
        GITHUB_RAW = "https://raw.githubusercontent.com/GGB638/ai-image-gan/main/version.txt"
        LOCAL_VERSION = "1.0.0"  # Текущая версия
        self.log("Проверка обновлений...")
        try:
            r = requests.get(GITHUB_RAW, timeout=10)
            r.raise_for_status()
            remote_version = r.text.strip()
            if remote_version != LOCAL_VERSION:
                self.log(f"Доступна новая версия: {remote_version} (текущая: {LOCAL_VERSION})")
                answer = messagebox.askyesno(
                    "Обновление доступно",
                    f"Доступна версия {remote_version}\n"
                    f"Текущая: {LOCAL_VERSION}\n\n"
                    f"Скачать обновление?\n"
                    f"(Файл заменится автоматически)")
                if answer:
                    self._download_update(remote_version)
            else:
                self.log(f"У тебя последняя версия: {LOCAL_VERSION} ✓")
                self.show_info("Обновления", f"У тебя последняя версия: {LOCAL_VERSION} ✓")
        except Exception as e:
            self.log(f"Проверка обновлений не удалась: {e}")
            self.show_info("Обновления", f"Не удалось проверить:\n{e}\n\nПроверь подключение к интернету.")

    def _download_update(self, version: str):
        GITHUB_PY = "https://raw.githubusercontent.com/GGB638/ai-image-gan/main/ai_image_gen.py"
        try:
            self.log(f"Скачиваю версию {version}...")
            r = requests.get(GITHUB_PY, timeout=60)
            r.raise_for_status()
            current_file = os.path.abspath(__file__)
            backup = current_file + ".backup"
            import shutil
            shutil.copy2(current_file, backup)
            with open(current_file, "w", encoding="utf-8") as f:
                f.write(r.text)
            self.log(f"✓ Обновление {version} установлено! Резервная копия: {backup}")
            self.show_info("Обновление установлено",
                           f"Версия {version} установлена!\n"
                           f"Резервная копия: {os.path.basename(backup)}\n\n"
                           f"Перезапусти программу для применения.")
        except Exception as e:
            self.log(f"Ошибка загрузки обновления: {e}")
            self.show_error("Ошибка", f"Не удалось скачать обновление:\n{e}")

    def _refresh_model_list(self):
        """Сканирует папку models/ и обновляет выпадающий список."""
        found = ["Авто (встроенная)"]
        if os.path.isdir(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                if f.lower().endswith(ALLOWED_MODEL_EXTENSIONS):
                    found.append(f)
        self.esrgan_model_combo["values"] = found
        if self.esrgan_model_var.get() not in found:
            self.esrgan_model_var.set("Авто (встроенная)")
        count = len(found) - 1
        self.log(f"Список моделей обновлён. Найдено в models/: {count} файлов.")

    def _draw_loss_graph(self):
        """Рисует график Loss D/G на canvas."""
        c = self.loss_canvas
        c.delete("all")
        w, h = 500, 180
        pad = 30
        c.create_text(pad, 10, text="Loss D/G", fill="#95A2BD", anchor="w", font=("Consolas", 8))
        c.create_text(pad + 120, 10, text="— D", fill="#4F7CFF", anchor="w", font=("Consolas", 8))
        c.create_text(pad + 160, 10, text="— G", fill="#FF6B6B", anchor="w", font=("Consolas", 8))
        # Оси
        c.create_line(pad, h - pad, w - 10, h - pad, fill="#2B3550")
        c.create_line(pad, h - pad, pad, 10, fill="#2B3550")

        if not self._loss_d_history:
            c.create_text(w // 2, h // 2, text="Ожидание данных...", fill="#95A2BD", font=("Consolas", 9))
            return

        max_val = max(max(self._loss_d_history + self._loss_g_history, default=1), 1)
        n = len(self._loss_d_history)
        def xp(i): return pad + int((i / max(n - 1, 1)) * (w - pad - 10))
        def yp(v): return h - pad - int((v / max_val) * (h - pad - 10))

        # Grid lines
        for val in [0.5, 1.0, 1.5, 2.0]:
            if val <= max_val:
                y = yp(val)
                c.create_line(pad, y, w - 10, y, fill="#1A2035", dash=(2, 4))
                c.create_text(pad - 2, y, text=f"{val:.1f}", fill="#4A5568", anchor="e", font=("Consolas", 7))

        # Loss D
        if len(self._loss_d_history) > 1:
            pts_d = [(xp(i), yp(min(v, max_val))) for i, v in enumerate(self._loss_d_history)]
            c.create_line(*[c for pt in pts_d for c in pt], fill="#4F7CFF", width=1, smooth=True)
        # Loss G
        if len(self._loss_g_history) > 1:
            pts_g = [(xp(i), yp(min(v, max_val))) for i, v in enumerate(self._loss_g_history)]
            c.create_line(*[c for pt in pts_g for c in pt], fill="#FF6B6B", width=1, smooth=True)

    def _push_loss_point(self, loss_d: float, loss_g: float):
        """Добавляет точку в историю и перерисовывает граф. Вызывать из training loop."""
        max_points = 300
        self._loss_d_history.append(loss_d)
        self._loss_g_history.append(loss_g)
        self._current_run_losses_d.append(loss_d)
        self._current_run_losses_g.append(loss_g)
        if len(self._loss_d_history) > max_points:
            self._loss_d_history = self._loss_d_history[-max_points:]
            self._loss_g_history = self._loss_g_history[-max_points:]
        self._ui_call(self._draw_loss_graph)

    def start_analyze_dataset(self):
        self.run_async(self._analyze_dataset_flow)

    def _analyze_dataset_flow(self):
        self.log("=== АНАЛИЗ ДАТАСЕТА ===")
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        if not files:
            self.log("Датасет пуст.")
            return
        widths, heights, ratios = [], [], []
        broken = 0
        for path in files:
            try:
                img = Image.open(path)
                w, h = img.size
                widths.append(w); heights.append(h)
                ratios.append(round(max(w, h) / max(min(w, h), 1), 2))
            except Exception:
                broken += 1
        total = len(files)
        portrait = sum(1 for r in ratios if r > 1.2)
        square = sum(1 for r in ratios if r <= 1.2)
        very_wide = sum(1 for r in ratios if r > 2.0)
        report = [
            f"Всего файлов: {total}",
            f"Битых файлов: {broken}",
            f"Разрешение: мин {min(widths)}x{min(heights)}, макс {max(widths)}x{max(heights)}, среднее {int(sum(widths)/len(widths))}x{int(sum(heights)/len(heights))}",
            f"Квадратных (~1:1): {square} ({100*square//total}%)",
            f"Портрет/пейзаж (>1.2): {portrait} ({100*portrait//total}%)",
            f"Очень вытянутых (>2:1): {very_wide} ({100*very_wide//total}%) — рекомендуется удалить",
            f"",
            f"Совет: {'Датасет однородный ✓' if very_wide < total * 0.1 else 'Много вытянутых — запусти Очистку по соотношению сторон'}",
        ]
        for line in report:
            self.log(line)
        self.show_info("Анализ датасета", "\n".join(report))

    def start_aspect_cleanup(self):
        self.run_async(self._aspect_cleanup_flow)

    def _aspect_cleanup_flow(self):
        self.log("=== ОЧИСТКА ПО СООТНОШЕНИЮ СТОРОН ===")
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        removed = 0
        for path in files:
            try:
                img = Image.open(path)
                w, h = img.size
                ratio = max(w, h) / max(min(w, h), 1)
                if ratio > 1.8:  # удаляем слишком вытянутые
                    os.remove(path)
                    removed += 1
            except Exception:
                continue
        self.log(f"Удалено вытянутых изображений (ratio > 1.8): {removed}")
        self.show_info("Очистка завершена", f"Удалено {removed} вытянутых изображений.\nОсталось: {self.count_dataset_images()}")

    def start_augment_dataset(self):
        self.run_async(self._augment_dataset_flow)

    def _augment_dataset_flow(self):
        self.log("=== АУГМЕНТАЦИЯ ДАТАСЕТА ===")
        files = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
                 if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)
                 and "_aug_" not in f]  # не аугментируем уже аугментированные
        created = 0
        for path in files:
            try:
                img = Image.open(path).convert("RGB")
                base = os.path.splitext(os.path.basename(path))[0]
                # Зеркало по горизонтали
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped.save(os.path.join(DATASET_DIR, f"{base}_aug_flip.jpg"), quality=95)
                created += 1
                # Поворот на 90° (только для квадратных/близких к квадрату)
                w, h = img.size
                if max(w, h) / max(min(w, h), 1) < 1.3:
                    rot = img.rotate(90, expand=True)
                    rot.save(os.path.join(DATASET_DIR, f"{base}_aug_rot90.jpg"), quality=95)
                    created += 1
            except Exception:
                continue
        self.log(f"Создано аугментированных копий: {created}. Итого в датасете: {self.count_dataset_images()}")
        self.show_info("Аугментация завершена", f"Создано {created} новых изображений.\nИтого в датасете: {self.count_dataset_images()}")

    def start_interpolation(self):
        self.run_async(self._interpolation_flow)

    def _interpolation_flow(self):
        self.log("=== LATENT SPACE ИНТЕРПОЛЯЦИЯ ===")
        ckpt = self.ckpt_var.get().strip() or self.newest_checkpoint_path()
        if not ckpt or not os.path.isfile(ckpt):
            raise ValueError("Чекпоинт не найден.")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        image_size = int(state.get("image_size", 64))
        latent_dim = int(state.get("latent_dim", LATENT_DIM))
        netG = Generator(image_size=image_size, latent_dim=latent_dim).to(self.device)
        gen_state = state.get("generator_ema_state_dict", state["generator_state_dict"])
        netG.load_state_dict(gen_state, strict=False)
        netG.eval()
        trunc = float(self.truncation_var.get())
        steps = 16
        seed = int(self.seed_var.get())
        if seed >= 0:
            torch.manual_seed(seed)
        z1 = torch.randn(1, latent_dim, 1, 1, device=self.device) * trunc
        z2 = torch.randn(1, latent_dim, 1, 1, device=self.device) * trunc
        frames = []
        with torch.no_grad():
            for i in range(steps):
                alpha = i / (steps - 1)
                z = z1 * (1 - alpha) + z2 * alpha
                img = netG(z).detach().cpu()
                frames.append(img)
        grid = torch.cat(frames, dim=0)
        out_path = os.path.join(OUTPUT_DIR, f"interpolation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        from torchvision import utils as tv_utils
        tv_utils.save_image(grid, out_path, normalize=True, nrow=8)
        self.log(f"Интерполяция сохранена: {out_path}")
        self.show_info("Готово", f"Сохранено {steps} кадров интерполяции:\n{out_path}")

    def _apply_upscale(self, img: Image.Image, mode: str) -> Image.Image:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        factor = 4 if "x4" in mode else 2
        if "Lanczos" in mode:
            from PIL import ImageFilter
            upscaled = img.resize((img.width * factor, img.height * factor), resample)
            return upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
        elif "Real-ESRGAN" in mode:
            return self._apply_realesrgan(img, factor)
        return img

    def _apply_realesrgan(self, img: Image.Image, scale: int) -> Image.Image:
        """Апскейл через Real-ESRGAN. Скачивает модель при первом использовании."""
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
        except ImportError:
            self.log("Real-ESRGAN не установлен. Устанавливаю...")
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "realesrgan", "basicsr"], check=True)
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

        selected = self.esrgan_model_var.get()
        mode = self.esrgan_mode_var.get()

        if selected != "Авто (встроенная)":
            # Пользователь выбрал свою модель из папки models/
            model_path = os.path.join(MODELS_DIR, selected)
            if not os.path.isfile(model_path):
                raise ValueError(f"Файл модели не найден: {model_path}")
            self.log(f"Real-ESRGAN: используется кастомная модель {selected}")
            # Пробуем угадать архитектуру по названию файла
            name_lower = selected.lower()
            if "anime" in name_lower or "6b" in name_lower:
                num_feat, num_block = 32, 6
            else:
                num_feat, num_block = 64, 23
        else:
            # Встроенная модель — скачиваем если нужно
            if mode == "Рисунок":
                model_name = "RealESRGAN_x4plus_anime_6B"
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
                num_feat, num_block = 32, 6
            else:
                model_name = "RealESRGAN_x4plus" if scale == 4 else "RealESRGAN_x2plus"
                model_url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
                num_feat, num_block = 64, 23
            model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
            if not os.path.isfile(model_path):
                self.log(f"Скачиваю модель {model_name}...")
                r = requests.get(model_url, timeout=180)
                with open(model_path, "wb") as f:
                    f.write(r.content)
                self.log("Модель скачана.")

        # Загрузка весов — поддержка .pth, .pt, .safetensors, .ckpt, .bin
        ext = os.path.splitext(model_path)[1].lower()
        if ext == ".safetensors":
            try:
                from safetensors.torch import load_file as sf_load
                weights = sf_load(model_path, device="cpu")
            except ImportError:
                self.log("Устанавливаю safetensors...")
                import subprocess, sys as _sys
                subprocess.run([_sys.executable, "-m", "pip", "install", "safetensors"], check=True)
                from safetensors.torch import load_file as sf_load
                weights = sf_load(model_path, device="cpu")
        else:
            weights = torch.load(model_path, map_location="cpu", weights_only=False)
            if "params_ema" in weights:
                weights = weights["params_ema"]
            elif "params" in weights:
                weights = weights["params"]

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat,
                        num_block=num_block, num_grow_ch=32, scale=scale)
        model.load_state_dict(weights, strict=False)
        upsampler = RealESRGANer(scale=scale, model_path=model_path, model=model,
                                  tile=256, tile_pad=10, pre_pad=0, half=False)
        import numpy as np
        img_np = np.array(img)
        out_np, _ = upsampler.enhance(img_np, outscale=scale)
        return Image.fromarray(out_np)

    def _update_image_preview_widget(self, label_widget, image_path: str, max_size=(260, 260)):
        if not image_path or not os.path.isfile(image_path):
            label_widget.configure(text="Нет картинки", image="")
            label_widget.image = None
            return
        try:
            img = Image.open(image_path).convert("RGB")
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img.thumbnail(max_size, resample)
            tk_img = ImageTk.PhotoImage(img)
            label_widget.configure(image=tk_img, text="")
            label_widget.image = tk_img
        except Exception:
            label_widget.configure(text="Не удалось показать картинку", image="")
            label_widget.image = None

    def refresh_generated_preview(self):
        path = os.path.join(OUTPUT_DIR, "generated.jpg")
        self._ui_call(self._update_image_preview_widget, self.gen_preview_label, path)

    def refresh_last_training_preview(self):
        previews = sorted(
            [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.startswith("preview_epoch_") and f.endswith(".jpg")],
            key=os.path.getmtime,
        )
        path = previews[-1] if previews else ""
        self._ui_call(self._update_image_preview_widget, self.train_preview_label, path)

    def _is_ad_like_url(self, url: str) -> bool:
        lowered = url.lower()
        if re.search(r"(^|[/_.\-])(ad|ads)([/_.\-]|$)", lowered):
            return True
        return any(k in lowered for k in AD_URL_KEYWORDS if k not in ("ad", "ads"))

    def _image_hash(self, image: Image.Image) -> str:
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()

    def _existing_dataset_hashes(self):
        hashes = set()
        for name in os.listdir(DATASET_DIR):
            if not name.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                continue
            path = os.path.join(DATASET_DIR, name)
            try:
                img = Image.open(path).convert("RGB")
                hashes.add(self._image_hash(img))
            except Exception:
                continue
        return hashes

    def cleanup_dataset(self):
        mode = self.download_filter_var.get().strip().lower() if hasattr(self, "download_filter_var") else "strict"
        min_side = 128 if mode == "strict" else 96
        max_ratio = 3.0 if mode == "strict" else 4.5

        removed_bad = 0
        removed_dup = 0
        kept_hashes = set()
        total = 0

        files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
        for name in files:
            total += 1
            path = os.path.join(DATASET_DIR, name)
            remove = False
            remove_reason = ""
            try:
                img_raw = Image.open(path)
                if getattr(img_raw, "is_animated", False):
                    remove = True
                    remove_reason = "bad"
                else:
                    image = img_raw.convert("RGB")
                    w, h = image.size
                    ratio = max(w / max(1, h), h / max(1, w))
                    if min(w, h) < min_side or ratio > max_ratio:
                        remove = True
                        remove_reason = "bad"
                    else:
                        hsh = self._image_hash(image)
                        if hsh in kept_hashes:
                            remove = True
                            remove_reason = "dup"
                        else:
                            kept_hashes.add(hsh)
            except Exception:
                remove = True
                remove_reason = "bad"

            if remove:
                try:
                    os.remove(path)
                    if remove_reason == "dup":
                        removed_dup += 1
                    else:
                        removed_bad += 1
                except Exception:
                    pass

        left = self.count_dataset_images()
        self.log(
            f"Dataset cleanup done ({mode}). Removed={removed_bad + removed_dup}, "
            f"duplicates≈{removed_dup}, bad≈{removed_bad}, left={left}"
        )
        self.show_info(self.msg("done_title"), f"Cleanup completed.\nLeft images: {left}")

    def log(self, message: str):
        if not self._is_main_thread():
            self.root.after(0, lambda: self.log(message))
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        # Дублируем в файл если включено
        try:
            if getattr(self, "file_log_var", None) and self.file_log_var.get():
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass

    def _start_activity_indicator(self):
        self._activity_tick = 0
        self.busy_bar.start(70)
        self._animate_status()

    def _stop_activity_indicator(self):
        if self._activity_after_id is not None:
            try:
                self.root.after_cancel(self._activity_after_id)
            except Exception:
                pass
            self._activity_after_id = None
        self.busy_bar.stop()
        self.set_status(self.msg("status_done"))

    def _animate_status(self):
        if not self.busy:
            return
        dots = "." * (self._activity_tick % 4)
        self.set_status(f"{self.msg('status_running')}{dots}")
        self._activity_tick += 1
        self._activity_after_id = self.root.after(500, self._animate_status)

    def run_async(self, fn):
        if self.busy:
            self.show_warning(self.msg("wait_title"), self.msg("running_task"))
            return

        self.busy = True
        self._start_activity_indicator()

        def wrap():
            try:
                fn()
            except Exception as e:
                self.log(f"ERROR: {e}")
                self.show_error(self.msg("error_title"), str(e))
            finally:
                self.busy = False
                self.root.after(0, self._stop_activity_indicator)

        threading.Thread(target=wrap, daemon=True).start()

    def start_download(self):
        self.run_async(lambda: self.download_images(show_popup=True))

    def start_cleanup_dataset(self):
        self.run_async(self.cleanup_dataset)

    def start_training(self):
        self.run_async(lambda: self.train_gan(show_popup=True))

    def start_resume_training(self):
        ckpt = self.resume_ckpt_var.get().strip()
        if not ckpt:
            ckpt = self.newest_checkpoint_path()
            if not ckpt:
                self.show_warning(self.msg("error_title"), "Не найден чекпоинт для дообучения.")
                return
            self.resume_ckpt_var.set(ckpt)
        self.run_async(lambda: self.train_gan(show_popup=True, resume_checkpoint=ckpt))

    def start_upscale_finetune(self):
        ckpt = self.resume_ckpt_var.get().strip()
        if not ckpt:
            ckpt = self.newest_checkpoint_path()
            if not ckpt:
                self.show_warning(self.msg("error_title"), "Не найден чекпоинт для дообучения.")
                return
            self.resume_ckpt_var.set(ckpt)
        self.run_async(lambda: self.train_gan(show_popup=True, resume_checkpoint=ckpt, allow_resolution_upgrade=True))

    def start_generation(self):
        self.run_async(lambda: self.generate_images(show_popup=True))

    def _is_memory_error(self, error: Exception) -> bool:
        text = str(error).lower()
        markers = [
            "out of memory",
            "not enough memory",
            "failed to allocate",
            "insufficient memory",
            "ran out of memory",
            "e_outofmemory",
        ]
        return any(m in text for m in markers)

    def _extract_image_urls(self, html: str, base_url: str):
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        seen = set()
        for img in soup.find_all("img"):
            candidates = []
            for attr in ("src", "data-src", "data-lazy-src"):
                v = img.get(attr)
                if v:
                    candidates.append(v)

            srcset = img.get("srcset")
            if srcset:
                parsed = []
                for p in srcset.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    chunks = p.split()
                    url_part = chunks[0]
                    score = 0
                    if len(chunks) > 1:
                        m = re.match(r"(\d+)(w|x)$", chunks[1].strip())
                        if m:
                            score = int(m.group(1))
                    parsed.append((score, url_part))
                parsed.sort(key=lambda x: x[0], reverse=True)
                candidates.extend([p[1] for p in parsed])

            for raw in candidates:
                u = urljoin(base_url, raw.strip())
                if not u.startswith("http"):
                    continue
                if self._is_ad_like_url(u):
                    continue
                if u in seen:
                    continue
                seen.add(u)
                urls.append(u)
        return urls

    def download_images(self, show_popup=True):
        url = self.url_var.get().strip()
        max_count = int(self.count_var.get())
        if not url:
            raise ValueError("Введите URL.")
        mode = self.download_filter_var.get().strip().lower() if hasattr(self, "download_filter_var") else "strict"
        min_side = 128 if mode == "strict" else 96
        max_ratio = 3.0 if mode == "strict" else 4.5

        self.log(f"Downloading image links from: {url} | filter={mode}")
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        candidates = self._extract_image_urls(resp.text, url)
        self.log(f"Found {len(candidates)} image URLs.")

        existing_hashes = self._existing_dataset_hashes()
        downloaded = 0
        skipped_small = 0
        skipped_ads = 0
        skipped_dups = 0
        skipped_format = 0
        for i, img_url in enumerate(candidates, start=1):
            if downloaded >= max_count:
                break
            try:
                if self._is_ad_like_url(img_url):
                    skipped_ads += 1
                    continue
                r = requests.get(img_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()

                img_raw = Image.open(io.BytesIO(r.content))
                if getattr(img_raw, "is_animated", False):
                    continue
                image = img_raw.convert("RGB")
                w, h = image.size
                if min(w, h) < min_side:
                    skipped_small += 1
                    continue
                ratio = max(w / max(1, h), h / max(1, w))
                if ratio > max_ratio:
                    skipped_ads += 1
                    continue

                hsh = self._image_hash(image)
                if hsh in existing_hashes:
                    skipped_dups += 1
                    continue
                existing_hashes.add(hsh)

                # Keep high quality and support common static formats.
                src_fmt = (img_raw.format or "").upper()
                if src_fmt in ("JPEG", "JPG"):
                    ext = ".jpg"
                    path = os.path.join(DATASET_DIR, f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
                    image.save(path, format="JPEG", quality=97, optimize=True)
                elif src_fmt == "PNG":
                    ext = ".png"
                    path = os.path.join(DATASET_DIR, f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
                    image.save(path, format="PNG", optimize=True)
                elif src_fmt == "WEBP":
                    ext = ".webp"
                    path = os.path.join(DATASET_DIR, f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
                    image.save(path, format="WEBP", quality=98, method=6)
                elif src_fmt in ("BMP", "TIFF", "TIF"):
                    ext = ".png"
                    path = os.path.join(DATASET_DIR, f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}")
                    image.save(path, format="PNG", optimize=True)
                else:
                    skipped_format += 1
                    continue
                downloaded += 1

                if downloaded % 10 == 0 or downloaded == 1:
                    self.log(f"Downloaded {downloaded}/{max_count}...")
                    self.set_status(f"Скачивание: {downloaded}/{max_count}...")
            except Exception:
                continue

            if i % 50 == 0:
                self.log(f"Processed {i} links, downloaded {downloaded}.")

        self.log(
            f"✅ Скачивание завершено! Сохранено {downloaded} изображений → dataset/ "
            f"(пропущено: мелкие={skipped_small}, реклама={skipped_ads}, дубли={skipped_dups}, формат={skipped_format})"
        )
        log_to_file(f"Download complete: {downloaded} images saved")
        self._ui_call(self._refresh_dataset_info)
        self.set_status(f"✅ Скачано {downloaded} картинок")
        # Всегда показываем уведомление — независимо от show_popup
        threading.Thread(target=_play_done_sound, daemon=True).start()
        if show_popup:
            self.show_info(self.msg("done_title"),
                           f"✅ Скачано: {downloaded} изображений\n"
                           f"Папка: dataset/\n\n"
                           f"Пропущено:\n"
                           f"  Мелкие (<min): {skipped_small}\n"
                           f"  Реклама: {skipped_ads}\n"
                           f"  Дубликаты: {skipped_dups}\n"
                           f"  Формат: {skipped_format}")
        else:
            self.log(f"✓ Автоскачивание: {downloaded} изображений готово → dataset/")

    def train_gan(self, show_popup=True, resume_checkpoint="", allow_resolution_upgrade=False):
        epochs = int(self.epochs_var.get())
        batch_size_initial = int(self.batch_var.get())
        image_size = int(self.size_var.get())

        if image_size not in (64, 128, 256):
            raise ValueError("Размер должен быть 64, 128 или 256.")
        if image_size == 256 and "CUDA" not in self.device_name:
            self.log("⚠ 256px без CUDA очень медленно. Рекомендуется 64/128 для DirectML.")
        if epochs < 8:
            self.log("⚠ Мало эпох — для видимого качества нужно 30+.")

        # Собираем изображения — рекурсивно из dataset/ и всех подпапок
        image_files = []
        for root, dirs, files in os.walk(DATASET_DIR):
            for f in files:
                if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, f))
        if not image_files:
            raise ValueError("В dataset/ нет изображений. Сначала скачайте картинки.")
        self.log(f"Датасет: {len(image_files)} изображений (включая подпапки)")

        # Улучшенный transform — резкость через RandomSharpness
        transform_list = [
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15)),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        ]
        # RandomSharpness — помогает против мыла (доступна в torchvision >= 0.10)
        try:
            transform_list.append(transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3))
        except AttributeError:
            pass
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        transform = transforms.Compose(transform_list)

        dataset = FlatImageDataset(image_files, transform=transform)
        if len(dataset) == 0:
            raise ValueError("Не удалось создать датасет. Проверьте изображения в dataset/.")

        batch_size = batch_size_initial
        while True:
            use_pin_memory = hasattr(self.device, 'type') and self.device.type == 'cuda'
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=use_pin_memory)
            if len(loader) == 0:
                raise ValueError("Слишком большой batch size для текущего количества изображений.")

            self.log(f"Training started: epochs={epochs}, batch={batch_size}, size={image_size}, device={self.device_name}")
            if "DirectML" in self.device_name and image_size == 128:
                self.log("Tip: for DirectML, size=64 is usually much more stable than 128.")
            self.log(f"DiffAugment: {'ON' if self.diffaug_var.get() else 'OFF'}")

            netG = Generator(image_size=image_size).to(self.device)
            netD = Discriminator(image_size=image_size).to(self.device)
            criterion = nn.BCELoss()
            # TTUR: slightly lower LR for D helps avoid discriminator overpower.
            base_lr_d = 0.00004
            base_lr_g = 0.0003
            optimD = optim.Adam(netD.parameters(), lr=base_lr_d, betas=(0.5, 0.999))
            optimG = optim.Adam(netG.parameters(), lr=base_lr_g, betas=(0.5, 0.999))
            schedulerD = optim.lr_scheduler.CosineAnnealingLR(optimD, T_max=epochs, eta_min=5e-6)
            schedulerG = optim.lr_scheduler.CosineAnnealingLR(optimG, T_max=epochs, eta_min=1e-5)

            # Gradient accumulation и другие настройки из вкладки Настройки
            grad_accum = max(1, int(getattr(self, "grad_accum_var", type("x", (), {"get": lambda s: 1})()).get()))
            warmup_epochs = int(getattr(self, "warmup_epochs_var", type("x", (), {"get": lambda s: 0})()).get())
            fid_every = int(getattr(self, "fid_every_var", type("x", (), {"get": lambda s: 0})()).get())
            early_stop_patience = int(getattr(self, "early_stop_var", type("x", (), {"get": lambda s: 0})()).get())
            use_fp16 = getattr(self, "fp16_var", None) and self.fp16_var.get()
            use_multi_gpu = (getattr(self, "multi_gpu_var", None) and self.multi_gpu_var.get()
                             and "CUDA" in self.device_name and torch.cuda.device_count() > 1)

            if grad_accum > 1:
                self.log(f"Gradient accumulation: {grad_accum}x (эффективный batch = {batch_size * grad_accum})")
            if warmup_epochs > 0:
                self.log(f"LR Warmup: {warmup_epochs} эпох")
            if use_multi_gpu:
                netG = torch.nn.DataParallel(netG)
                netD = torch.nn.DataParallel(netD)
                self.log(f"Multi-GPU: {torch.cuda.device_count()} GPU")

            # AMP: только CUDA, на DirectML не работает надёжно
            use_amp = "CUDA" in self.device_name and torch.cuda.is_available() and use_fp16
            try:
                if use_amp:
                    scaler = torch.amp.GradScaler("cuda", enabled=True)
                else:
                    scaler = torch.amp.GradScaler("cpu", enabled=False)
            except Exception:
                try:
                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                except Exception:
                    scaler = None
                use_amp = False
            if use_amp:
                self.log("AMP FP16 включён (CUDA).")
            amp_device = "cuda" if use_amp else "cpu"

            early_stop_counter = 0
            best_fid_score = float("inf")

            if resume_checkpoint:
                if not os.path.isfile(resume_checkpoint):
                    raise ValueError(f"Checkpoint not found: {resume_checkpoint}")
                state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                ckpt_size = int(state.get("image_size", image_size))
                if ckpt_size != image_size:
                    if allow_resolution_upgrade:
                        self.log(
                            f"Resolution upgrade fine-tune: {ckpt_size} -> {image_size}. "
                            "Loading compatible layers only."
                        )
                        g_state = state.get("generator_ema_state_dict", state["generator_state_dict"])
                        g_loaded = self._load_partial_state(netG, g_state)
                        d_loaded = self._load_partial_state(netD, state.get("discriminator_state_dict", {}))
                        self.log(f"Partial load complete: G layers={g_loaded}, D layers={d_loaded}.")
                    else:
                        self.log(f"Resume checkpoint size is {ckpt_size}. Overriding current size {image_size} -> {ckpt_size}.")
                        image_size = ckpt_size
                        self.size_var.set(image_size)
                        # Restart current loop with corrected size/dataset transform.
                        transform = transforms.Compose(
                            [
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]
                        )
                        dataset = FlatImageDataset(image_files, transform=transform)
                        continue
                else:
                    netG.load_state_dict(state["generator_state_dict"], strict=False)
                    if "discriminator_state_dict" in state:
                        netD.load_state_dict(state["discriminator_state_dict"])
                    if "optimizer_g_state_dict" in state:
                        optimG.load_state_dict(state["optimizer_g_state_dict"])
                    if "optimizer_d_state_dict" in state:
                        optimD.load_state_dict(state["optimizer_d_state_dict"])
                    self.log(f"Resumed training from checkpoint: {resume_checkpoint}")
            else:
                netG.apply(weights_init)
                netD.apply(weights_init)

            # EMA generator gives smoother previews/checkpoints for beginners.
            emaG = copy.deepcopy(netG).to(self.device)
            if resume_checkpoint and os.path.isfile(resume_checkpoint):
                state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                if "generator_ema_state_dict" in state:
                    emaG.load_state_dict(state["generator_ema_state_dict"], strict=False)
            for p in emaG.parameters():
                p.requires_grad_(False)

            fixed_noise = torch.randn(16, LATENT_DIM, 1, 1, device=self.device)
            real_label = 1.0
            fake_label = 0.0
            best_balance_score = float("inf")

            # Перцептивный loss — ленивая инициализация
            use_perceptual = getattr(self, "perceptual_var", None) and self.perceptual_var.get()
            vgg_model_path = ""
            if use_perceptual:
                selected_vgg = getattr(self, "vgg_model_var", None)
                if selected_vgg and selected_vgg.get() != "Авто (VGG16 torchvision ~528MB)":
                    vgg_model_path = os.path.join(VGG_MODELS_DIR, selected_vgg.get())
            vgg_loss_fn = VGGPerceptualLoss(self.device, vgg_model_path) if use_perceptual else None
            if use_perceptual:
                model_name = os.path.basename(vgg_model_path) if vgg_model_path else "VGG16 авто"
                self.log(f"Перцептивный loss включён: {model_name}")
            best_ckpt_path = ""

            try:
                for epoch in range(epochs):
                    epoch_start = time.time()

                    # Warmup: линейно увеличиваем LR от 0 до base за warmup_epochs
                    if warmup_epochs > 0 and epoch < warmup_epochs:
                        warmup_factor = (epoch + 1) / warmup_epochs
                        for g in optimD.param_groups: g["lr"] = base_lr_d * warmup_factor
                        for g in optimG.param_groups: g["lr"] = base_lr_g * warmup_factor
                        self.log(f"Warmup LR: {warmup_factor:.2f}x")

                    self.log(f"Epoch {epoch + 1}/{epochs} started...")
                    real_preview_batch = None
                    epoch_loss_d_sum = 0.0
                    epoch_loss_g_sum = 0.0
                    epoch_steps = 0
                    optimD.zero_grad()
                    optimG.zero_grad()
                    accum_step = 0
                    for step, batch in enumerate(loader):
                        real = batch[0].to(self.device)
                        b_size = real.size(0)
                        if real_preview_batch is None:
                            real_preview_batch = real[:8].detach().cpu()
                        noise_sigma = max(0.0, 0.05 * (1.0 - (epoch / max(1, epochs - 1))))

                        netD.zero_grad()
                        # Small label smoothing/noise stabilizes GAN training.
                        label_real = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                        label_real = label_real * 0.95 + torch.rand_like(label_real) * 0.05  # [0.95, 1.0]
                        real_input = diff_augment(real) if self.diffaug_var.get() else real
                        if noise_sigma > 0:
                            real_input = torch.clamp(real_input + torch.randn_like(real_input) * noise_sigma, -1.0, 1.0)
                        amp_ctx = (lambda: torch.amp.autocast(amp_device)) if use_amp else nullcontext
                        with amp_ctx():
                            out_real = netD(real_input)
                            loss_real = criterion(out_real, label_real)
                        if use_amp:
                            scaler.scale(loss_real).backward()
                        else:
                            loss_real.backward()

                        noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=self.device)
                        fake = netG(noise)
                        label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=self.device)
                        label_fake = torch.rand_like(label_fake) * 0.05  # [0.0, 0.05]
                        fake_detached_input = diff_augment(fake.detach()) if self.diffaug_var.get() else fake.detach()
                        if noise_sigma > 0:
                            fake_detached_input = torch.clamp(
                                fake_detached_input + torch.randn_like(fake_detached_input) * noise_sigma, -1.0, 1.0
                            )
                        with amp_ctx():
                            out_fake = netD(fake_detached_input)
                            loss_fake = criterion(out_fake, label_fake)
                        if use_amp:
                            scaler.scale(loss_fake).backward()
                        else:
                            loss_fake.backward()
                        loss_d = loss_real + loss_fake
                        if use_amp:
                            scaler.unscale_(optimD)
                        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
                        if use_amp:
                            scaler.step(optimD)
                        else:
                            optimD.step()

                        # Адаптивное кол-во обновлений G:
                        # если D слишком силён (loss < 0.4) — обновляем G дважды,
                        # иначе — один раз. Это предотвращает деградацию G.
                        g_steps = 2 if loss_d.item() < 0.4 else 1
                        for _g_iter in range(g_steps):
                            netG.zero_grad()
                            noise_g = torch.randn(b_size, LATENT_DIM, 1, 1, device=self.device) if _g_iter == 1 else noise
                            fake_g = netG(noise_g) if _g_iter == 1 else fake
                            label_gen = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                            label_gen = label_gen * 0.95 + torch.rand_like(label_gen) * 0.05  # [0.95, 1.0]
                            fake_input = diff_augment(fake_g) if self.diffaug_var.get() else fake_g
                            if noise_sigma > 0:
                                fake_input = torch.clamp(fake_input + torch.randn_like(fake_input) * noise_sigma, -1.0, 1.0)
                            with amp_ctx():
                                out_gen = netD(fake_input)
                                loss_g_adv = criterion(out_gen, label_gen)
                                loss_g_div = diversity_regularizer(fake_g)
                                loss_g = loss_g_adv + 0.15 * loss_g_div
                                # Feature matching loss — детали без VGG
                                try:
                                    fm_loss = feature_matching_loss(netD, real[:fake_g.size(0)], fake_g)
                                    loss_g = loss_g + 0.1 * fm_loss
                                except Exception:
                                    pass
                                # Перцептивный loss если включён
                                if vgg_loss_fn is not None:
                                    try:
                                        perc = vgg_loss_fn(fake_g, real[:fake_g.size(0)])
                                        loss_g = loss_g + 0.1 * perc
                                    except Exception:
                                        pass
                            if use_amp:
                                scaler.scale(loss_g).backward()
                                scaler.unscale_(optimG)
                            else:
                                loss_g.backward()
                            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
                            if use_amp:
                                scaler.step(optimG)
                                scaler.update()
                            else:
                                optimG.step()
                            with torch.no_grad():
                                ema_decay = float(self.ema_decay_var.get())
                                for p_ema, p in zip(emaG.parameters(), netG.parameters()):
                                    p_ema.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

                        epoch_loss_d_sum += loss_d.item()
                        epoch_loss_g_sum += loss_g.item()
                        epoch_steps += 1
                        self._push_loss_point(loss_d.item(), loss_g.item())
                        self.log(
                                f"Epoch {epoch + 1}/{epochs} | Step {step + 1}/{len(loader)} | "
                                f"Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f} "
                                f"(adv {loss_g_adv.item():.4f}, div {loss_g_div.item():.4f})"
                            )

                    with torch.no_grad():
                        preview = emaG(fixed_noise).detach().cpu()
                        # Save a beginner-friendly panel: real examples + generated examples.
                        if real_preview_batch is not None:
                            panel = torch.cat([real_preview_batch, preview[:8]], dim=0)
                        else:
                            panel = preview
                        preview_path = os.path.join(OUTPUT_DIR, f"preview_epoch_{epoch + 1}.jpg")
                        utils.save_image(panel, preview_path, normalize=True, nrow=4)
                        self.refresh_last_training_preview()

                    epoch_seconds = time.time() - epoch_start
                    schedulerD.step()
                    schedulerG.step()

                    # Считаем средний баланс эпохи — идеал: Loss D ~0.5, Loss G ~1.5
                    collapse_detected = False
                    if epoch_steps > 0:
                        avg_d = epoch_loss_d_sum / epoch_steps
                        avg_g = epoch_loss_g_sum / epoch_steps
                        avg_div = sum(
                            loss_g_div.item() for _ in [1]  # последнее значение div за эпоху
                        ) / 1
                        balance_score = abs(avg_d - 0.5) + abs(avg_g - 1.5)
                        self.log(f"Epoch {epoch + 1} avg: Loss D={avg_d:.3f}, Loss G={avg_g:.3f}, balance={balance_score:.3f}")

                        # --- Детектор коллапса ---
                        # Признаки коллапса: D слишком сильный (avg_d < 0.25)
                        # или G слишком слабый (avg_g > 5.0)
                        # или div упал почти в ноль (< 0.03)
                        if epoch >= 5 and best_ckpt_path:
                            if avg_d < 0.25 or avg_g > 5.0 or loss_g_div.item() < 0.03:
                                collapse_detected = True
                                reason = []
                                if avg_d < 0.25:
                                    reason.append(f"Loss D слишком низкий ({avg_d:.3f})")
                                if avg_g > 5.0:
                                    reason.append(f"Loss G слишком высокий ({avg_g:.3f})")
                                if loss_g_div.item() < 0.03:
                                    reason.append(f"div близок к нулю ({loss_g_div.item():.4f})")
                                self.log(f"⚠️ КОЛЛАПС ОБНАРУЖЕН на эпохе {epoch + 1}: {', '.join(reason)}")
                                self.log(f"↩️ Откат к лучшему чекпоинту: {os.path.basename(best_ckpt_path)}")
                                state = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
                                netG.load_state_dict(state["generator_state_dict"], strict=False)
                                netD.load_state_dict(state["discriminator_state_dict"])
                                emaG.load_state_dict(state["generator_ema_state_dict"], strict=False)
                                # Перезапускаем оптимайзеры с более низким LR чтобы не повторить коллапс
                                for g in optimG.param_groups:
                                    g["lr"] = g["lr"] * 0.5
                                for g in optimD.param_groups:
                                    g["lr"] = g["lr"] * 0.5
                                self.log(f"✓ Откат выполнен. LR снижен вдвое для стабилизации.")

                        if not collapse_detected and balance_score < best_balance_score and epoch >= 5:
                            best_balance_score = balance_score
                            best_ckpt_name = f"gan_{image_size}_BEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                            best_ckpt_path = os.path.join(CHECKPOINT_DIR, best_ckpt_name)
                            torch.save(
                                {
                                    "generator_state_dict": netG.state_dict(),
                                    "discriminator_state_dict": netD.state_dict(),
                                    "generator_ema_state_dict": emaG.state_dict(),
                                    "optimizer_g_state_dict": optimG.state_dict(),
                                    "optimizer_d_state_dict": optimD.state_dict(),
                                    "image_size": image_size,
                                    "latent_dim": LATENT_DIM,
                                },
                                best_ckpt_path,
                            )
                            self.log(f"✓ Лучший чекпоинт сохранён (balance={balance_score:.3f}): {best_ckpt_name}")

                    remain = max(0.0, (epochs - (epoch + 1)) * epoch_seconds)
                    progress_pct = (epoch + 1) / epochs * 100
                    eta_str = f"ETA: {int(remain//60)}м {int(remain%60)}с" if remain > 0 else "Готово!"
                    self._ui_call(lambda p=progress_pct, e=eta_str: (
                        self.train_progress_var.set(p),
                        self.train_progress_label.configure(text=f"{p:.0f}%  {e}")
                    ) if hasattr(self, 'train_progress_var') else None)
                    self.log(
                        f"Epoch {epoch + 1}/{epochs} completed in {epoch_seconds:.1f}s. "
                        f"ETA: ~{remain / 60:.1f} min. Preview saved."
                    )
                    self.log("Preview guide: top rows are REAL dataset images, bottom rows are GENERATED images.")

                    # Автосохранение чекпоинта каждые 5 эпох
                    if (epoch + 1) % 5 == 0:
                        auto_ckpt_name = f"gan_{image_size}_ep{epoch + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                        auto_ckpt_path = os.path.join(CHECKPOINT_DIR, auto_ckpt_name)
                        torch.save(
                            {
                                "generator_state_dict": netG.state_dict(),
                                "discriminator_state_dict": netD.state_dict(),
                                "generator_ema_state_dict": emaG.state_dict(),
                                "optimizer_g_state_dict": optimG.state_dict(),
                                "optimizer_d_state_dict": optimD.state_dict(),
                                "image_size": image_size,
                                "latent_dim": LATENT_DIM,
                            },
                            auto_ckpt_path,
                        )
                        self.latest_checkpoint = auto_ckpt_path
                        self.ckpt_var.set(auto_ckpt_path)
                        self.log(f"Auto-checkpoint saved: epoch {epoch + 1}")
                break
            except RuntimeError as e:
                if self._is_memory_error(e) and batch_size > 1:
                    new_batch = max(1, batch_size // 2)
                    self.batch_var.set(new_batch)
                    self.log(f"Memory issue detected. Reducing batch: {batch_size} -> {new_batch} and retrying...")
                    batch_size = new_batch
                    if hasattr(torch.cuda, "empty_cache"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                raise

        checkpoint_name = f"gan_{image_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        torch.save(
            {
                "generator_state_dict": netG.state_dict(),
                "discriminator_state_dict": netD.state_dict(),
                "generator_ema_state_dict": emaG.state_dict(),
                "optimizer_g_state_dict": optimG.state_dict(),
                "optimizer_d_state_dict": optimD.state_dict(),
                "image_size": image_size,
                "latent_dim": LATENT_DIM,
            },
            checkpoint_path,
        )
        self.latest_checkpoint = checkpoint_path
        self.ckpt_var.set(checkpoint_path)

        self.log(f"Training complete. Checkpoint saved: {checkpoint_path}")
        log_to_file(f"Training complete: {checkpoint_path}")

        # Сохраняем запуск в историю
        try:
            avg_d = sum(self._current_run_losses_d[-50:]) / max(len(self._current_run_losses_d[-50:]), 1)
            avg_g = sum(self._current_run_losses_g[-50:]) / max(len(self._current_run_losses_g[-50:]), 1)
            self._save_training_record(
                date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                size=image_size,
                epochs=epochs,
                batch=batch_size,
                final_loss_d=round(avg_d, 4),
                final_loss_g=round(avg_g, 4),
                checkpoint=checkpoint_path,
            )
            if hasattr(self, "_history_tree"):
                self._ui_call(self._refresh_history)
        except Exception:
            pass

        # Звуковое уведомление
        threading.Thread(target=_play_done_sound, daemon=True).start()

        if show_popup:
            self.show_info(self.msg("done_title"), self.msg("train_done", path=checkpoint_path))

    def generate_images(self, show_popup=True):
        ckpt = self.ckpt_var.get().strip()
        gen_count = int(self.gen_count_var.get())
        gen_count = max(1, min(64, gen_count))
        trunc = float(self.truncation_var.get())
        seed = int(self.seed_var.get())
        upscale_mode = self.upscale_var.get()

        if not ckpt:
            if self.latest_checkpoint:
                ckpt = self.latest_checkpoint
                self.ckpt_var.set(ckpt)
            else:
                ckpts = sorted(
                    [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
                    key=os.path.getmtime,
                )
                if not ckpts:
                    raise ValueError("Не найден чекпоинт. Сначала обучите модель.")
                ckpt = ckpts[-1]
                self.ckpt_var.set(ckpt)

        if not os.path.isfile(ckpt):
            raise ValueError(f"Файл чекпоинта не найден: {ckpt}")

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        image_size = int(state.get("image_size", 64))
        latent_dim = int(state.get("latent_dim", LATENT_DIM))

        netG = Generator(image_size=image_size, latent_dim=latent_dim).to(self.device)
        gen_state = state.get("generator_ema_state_dict", state["generator_state_dict"])
        netG.load_state_dict(gen_state, strict=False)
        netG.eval()

        if seed >= 0:
            torch.manual_seed(seed)
            self.log(f"Seed: {seed}")

        with torch.no_grad():
            noise = torch.randn(gen_count, latent_dim, 1, 1, device=self.device) * trunc
            fake = netG(noise).detach().cpu()

        out_path = os.path.join(OUTPUT_DIR, "generated.jpg")
        utils.save_image(fake, out_path, normalize=True, nrow=min(8, gen_count))

        # Сохраняем каждую картинку отдельно + апскейл если выбран
        for i in range(gen_count):
            single_path = os.path.join(OUTPUT_DIR, f"generated_{i + 1:03d}.jpg")
            utils.save_image(fake[i : i + 1], single_path, normalize=True, nrow=1)
            if upscale_mode != "Нет":
                try:
                    img = Image.open(single_path).convert("RGB")
                    upscaled = self._apply_upscale(img, upscale_mode)
                    up_path = os.path.join(OUTPUT_DIR, f"generated_{i + 1:03d}_upscaled.jpg")
                    upscaled.save(up_path, quality=97)
                except Exception as e:
                    self.log(f"Апскейл {i+1} не удался: {e}")

        self.log(f"Сгенерировано {gen_count} картинок. Truncation={trunc:.2f}. Grid: {out_path}")
        if upscale_mode != "Нет":
            self.log(f"Апскейл применён: {upscale_mode}")
        if gen_count > 1:
            self.log(f"Отдельные файлы: generated_001.jpg ... generated_{gen_count:03d}.jpg")
        self.refresh_generated_preview()
        if show_popup:
            self.show_info(self.msg("done_title"), self.msg("gen_done", path=out_path))



class TermsDialog:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.accepted = False
        self.var_accept = tk.BooleanVar(value=False)
        self.var_confirm_illegal = tk.BooleanVar(value=False)
        self.var_age = tk.BooleanVar(value=False)
        self._build()

    def _build(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Условия использования")
        dlg.geometry("760x520")
        dlg.configure(bg="#141821")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", self._close_decline)
        self.dlg = dlg

        container = ttk.Frame(dlg, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Условия использования", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        text = tk.Text(container, wrap="word", height=20)
        text.pack(fill="both", expand=True, pady=(0, 8))
        text.configure(bg="#0F1320", fg="#D5DEF5", insertbackground="#D5DEF5", selectbackground="#2B3550")
        text.insert(
            "1.0",
            (
                "Перед использованием подтвердите согласие:\n\n"
                "1) Пользователь самостоятельно выбирает и загружает датасет.\n"
                "2) Пользователь несет полную ответственность за законность и этичность контента,\n"
                "   который он скачивает, хранит, обучает и генерирует.\n"
                "3) Запрещено использовать программу для незаконного, вредоносного или нарушающего права контента.\n"
                "   Контент сексуального характера с участием несовершеннолетних строго запрещен и незаконен.\n"
                "4) Автор программы и распространитель не несут ответственности за действия пользователя,\n"
                "   содержание датасета, результаты генерации и любые последствия использования.\n"
                "5) Любая юридическая ответственность за незаконный контент лежит исключительно на пользователе,\n"
                "   который загружает/обучает/генерирует такой контент.\n"
                "6) Используя программу, вы подтверждаете, что соблюдаете законы вашей страны и правила площадок,\n"
                "   откуда берете изображения.\n\n"
                "Если не согласны с условиями — закройте приложение."
            ),
        )
        text.configure(state="disabled")

        check = ttk.Checkbutton(
            container,
            text="Я прочитал(а), понимаю и принимаю условия использования.",
            variable=self.var_accept,
            command=self._toggle_accept,
        )
        check.pack(anchor="w", pady=(0, 6))

        check_age = ttk.Checkbutton(
            container,
            text="Я подтверждаю, что мне исполнилось 18 лет (или возраст совершеннолетия в моей стране).",
            variable=self.var_age,
            command=self._toggle_accept,
        )
        check_age.pack(anchor="w", pady=(0, 6))

        check2 = ttk.Checkbutton(
            container,
            text=(
                "Я подтверждаю, что НЕ буду использовать программу для незаконного контента "
                "(включая любой сексуальный контент с несовершеннолетними)."
            ),
            variable=self.var_confirm_illegal,
            command=self._toggle_accept,
        )
        check2.pack(anchor="w", pady=(0, 8))

        btns = ttk.Frame(container)
        btns.pack(fill="x")
        ttk.Button(btns, text="Отказ", command=self._close_decline).pack(side="right", padx=(8, 0))
        self.btn_accept = ttk.Button(btns, text="Принять и продолжить", command=self._close_accept, state="disabled")
        self.btn_accept.pack(side="right")

    def _toggle_accept(self):
        ok = self.var_accept.get() and self.var_confirm_illegal.get() and self.var_age.get()
        self.btn_accept.configure(state="normal" if ok else "disabled")

    def _write_acceptance_receipt(self):
        record = {
            "accepted_at_utc": datetime.utcnow().isoformat() + "Z",
            "user": getpass.getuser(),
            "machine": platform.node(),
            "platform": platform.platform(),
            "app": "AI Image Generator (GAN)",
            "terms_version": "2026-04-06",
            "accepted": True,
            "confirmed_adult_18_plus": True,
            "confirmed_no_illegal_content": True,
            "notice": "User bears full legal responsibility for illegal content.",
        }
        try:
            with open(LEGAL_ACCEPTANCE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _close_accept(self):
        self._write_acceptance_receipt()
        self.accepted = True
        self.dlg.destroy()

    def _close_decline(self):
        self.accepted = False
        self.dlg.destroy()

    def show(self) -> bool:
        self.root.wait_window(self.dlg)
        return self.accepted



def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    root.configure(bg="#141821")

    # Dark UI theme tuned for readability.
    style.configure(".", background="#141821", foreground="#E6EAF2")
    style.configure("TFrame", background="#141821")
    style.configure("TLabel", background="#141821", foreground="#E6EAF2")
    style.configure("Hint.TLabel", background="#141821", foreground="#95A2BD")
    style.configure("Title.TLabel", background="#141821", foreground="#F3F6FF", font=("Segoe UI", 16, "bold"))
    style.configure("TLabelframe", background="#141821", foreground="#E6EAF2")
    style.configure("TLabelframe.Label", background="#141821", foreground="#B8C5E0")
    style.configure("TNotebook", background="#141821", borderwidth=0)
    style.configure("TNotebook.Tab", background="#202736", foreground="#DCE4F8", padding=(12, 6))
    style.map("TNotebook.Tab", background=[("selected", "#2B3550")], foreground=[("selected", "#FFFFFF")])
    style.configure("TEntry", fieldbackground="#202736", foreground="#EAF1FF")
    style.configure("TSpinbox", fieldbackground="#202736", foreground="#EAF1FF")
    style.configure("TCombobox", fieldbackground="#202736", foreground="#EAF1FF")
    style.configure("TButton", background="#2A3346", foreground="#EAF1FF")
    style.map("TButton", background=[("active", "#36445E")])
    style.configure("Primary.TButton", background="#4F7CFF", foreground="#FFFFFF", padding=(12, 6))
    style.map("Primary.TButton", background=[("active", "#6A8FFF")])

    terms = TermsDialog(root)
    if not terms.show():
        root.destroy()
        return

    ImageGANApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
