==============================================================================
   AI Image Generator (GAN)
   Copyright (C) 2026 Ivan Nedostup (GGB_638), Kryvyi Rih, Ukraine.
   Distributed under GNU General Public License v3 (GNU GPL v3).
   
   This program comes with ABSOLUTELY NO WARRANTY.
   This is free software, and you are welcome to redistribute it
   under the conditions of GNU GPL v3. See the LICENSE file for details.
==============================================================================

Если программа не запускается,
 скачайте и установите Microsoft Visual C++ 2019/2022 по этой ссылке: https://aka.ms/vs/16/release/vc_redist.x64.exe

ЧТО ЭТО?
---------
Программа для обучения генеративно-состязательной нейросети (GAN) на
собственных изображениях и генерации новых картинок на основе выученных
паттернов. Создана с целью сделать обучение ИИ доступным для всех.

## Хештеги / Tags
#AI #GAN #ImageGeneration #DeepLearning #MachineLearning #NeuralNetwork #Python #PyTorch #Desktop #Windows #OpenSource #GPL3

СИСТЕМНЫЕ ТРЕБОВАНИЯ
--------------------
- Windows 10 / 11 (64-bit)
- Python 3.10 или 3.11
- GPU: любая карта с поддержкой DirectX 12 (AMD, Intel, NVIDIA)
  * Для NVIDIA рекомендуется CUDA — обучение значительно быстрее
  * Для AMD/Intel используется DirectML (автоматически)
- Оперативная память: минимум 8 ГБ
- Место на диске: зависит от датасета и чекпоинтов


УСТАНОВКА
---------
1. Установи Python 3.10 или 3.11 с сайта python.org
   (при установке поставь галочку "Add Python to PATH")

2. Установи зависимости. Открой командную строку в папке с программой и выполни:

   pip install torch torchvision requests beautifulsoup4 pillow

   Если у тебя AMD или Intel GPU, дополнительно:
   pip install torch-directml

3. Запусти программу:
   python ai_image_gen.py


КАК ПОЛЬЗОВАТЬСЯ
----------------

ШАГ 1 — СКАЧИВАНИЕ ДАТАСЕТА
  - Перейди на вкладку "Скачивание"
  - Вставь URL сайта с изображениями (например: https://unsplash.com)
  - Укажи количество картинок (рекомендуется 1000+, идеально 3000+)
  - Нажми "Скачать"
  - Картинки сохранятся в папку: dataset/

  Совет: чем больше и разнообразнее датасет — тем лучше результат.
  Можно также вручную скопировать свои изображения в папку dataset/.

ШАГ 2 — ОБУЧЕНИЕ
  - Перейди на вкладку "Обучение"
  - Выбери разрешение:
      64px  — быстро, подходит для слабых GPU
      128px — хорошее качество, нужно больше времени и памяти
      256px — высокое качество, только для мощных NVIDIA GPU
  - Укажи количество эпох (рекомендуется 50–100)
  - Нажми "Быстрый старт обучения" или "Полный авто-режим"
  - В логах внизу будет отображаться прогресс каждого шага
  - Превью генерируется после каждой эпохи в папку: output/
  - Лучший чекпоинт сохраняется автоматически (файл с "BEST" в названии)

  Что смотреть в логах:
    Loss D ~0.5  — дискриминатор в балансе (хорошо)
    Loss G ~1.5  — генератор работает нормально (хорошо)
    div > 0.1    — нет mode collapse (хорошо)

ШАГ 3 — ГЕНЕРАЦИЯ
  - Перейди на вкладку "Генерация"
  - Выбери чекпоинт (рекомендуется файл с "BEST" в названии)
  - Укажи количество изображений
  - Нажми "Генерировать"
  - Результаты сохранятся в папку: output/


РАБОЧИЕ ПАПКИ
-------------
  dataset/      — твои обучающие изображения
  checkpoints/  — сохранённые модели (*.pt файлы)
  output/       — сгенерированные картинки и превью эпох


СОВЕТЫ ДЛЯ ЛУЧШЕГО РЕЗУЛЬТАТА
------------------------------
- Используй минимум 1000 картинок, идеально 3000+
- Картинки должны быть тематически похожи (лица, пейзажи и т.д.)
- Начинай с разрешения 64px, переходи на 128px после успешного обучения
- Используй прогрессивное обучение: сначала 64px, затем дообучи на 128px
- Следи за превью — если качество деградирует, используй чекпоинт "BEST"
- DiffAugment рекомендуется держать включённым при небольших датасетах


ВОЗМОЖНЫЕ ПРОБЛЕМЫ
------------------
Программа зависает на старте:
  - Убедись что все зависимости установлены
  - Попробуй запустить через командную строку для просмотра ошибок

Все изображения одинаковые (mode collapse):
  - Останови обучение и загрузи лучший чекпоинт "BEST"
  - Или начни обучение заново с нуля

Ошибка памяти (OOM):
  - Уменьши batch size в настройках
  - Перейди на меньшее разрешение (64px вместо 128px)

Обучение очень медленное:
  - На AMD/Intel это нормально — DirectML медленнее CUDA
  - Уменьши разрешение или batch size


ЛИЦЕНЗИЯ
--------
Эта программа является свободным программным обеспечением.
Вы можете распространять и/или изменять её согласно условиям
GNU General Public License версии 3, опубликованной
Free Software Foundation.

Полный текст лицензии — в файле LICENSE.
Онлайн: https://www.gnu.org/licenses/gpl-3.0.html

ОТКАЗ ОТ ГАРАНТИЙ:
Программа распространяется В НАДЕЖДЕ, ЧТО ОНА БУДЕТ ПОЛЕЗНОЙ,
НО БЕЗ КАКИХ-ЛИБО ГАРАНТИЙ, даже без подразумеваемой гарантии
ПРИГОДНОСТИ ДЛЯ КОНКРЕТНОЙ ЦЕЛИ. Подробнее — в GNU GPL v3.


КОНТАКТЫ
--------
Автор: Ivan Nedostup (GGB_638)
Местонахождение: Кривой Рог, Украина
Год: 2026

==============================================================================
# AI Image Generator (GAN)

Easy desktop app for beginners:
- Download images from a website
- Train GAN on your dataset
- Generate new images
- Language switch: Russian / English (from app UI)

## Tags
#AI #GAN #ImageGeneration #DeepLearning #MachineLearning #NeuralNetwork #Python #PyTorch #Desktop #Windows #OpenSource #GPL3

## Quick Start (Windows)

1. Install Python 3.11+ from [python.org](https://www.python.org/downloads/)  
   During install enable **Add Python to PATH**.
2. Run `install_and_run.bat`
3. In app:
   - Tab 1: download images to `dataset/`
   - Tab 2: train model (use **Auto** profile for first run)
   - Tab 3: generate output image

## Where data is stored

- You do **not** need to create folders manually in GitHub.
- The app creates all required folders automatically on first run.
- For Python run and EXE run, data is saved next to the app in:
  - `dataset/`
  - `checkpoints/`
  - `output/`

## Build EXE

Run:

`build_exe.bat`

Result:

`dist/AI_Image_Generator.exe`

## GPU Notes

- NVIDIA: CUDA can work if your PyTorch install supports it
- AMD/Intel: install `torch-directml` for DirectML acceleration
- CPU only: works, but slower

## Dependencies

See `requirements.txt`.

## Legal / Responsibility

- User is fully responsible for any dataset and generated content.
- Illegal content is strictly prohibited.
- Any sexual content involving minors is strictly prohibited and illegal.
- Any adult (18+) sexual content is strictly prohibited.
- The author/distributor is not responsible for user actions or legal consequences.
- On startup, user must explicitly accept terms and confirm non-use for illegal content.
- Acceptance events are logged locally in `legal_acceptance_log.jsonl`.
- Full legal text: `TERMS_OF_USE.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Content policy: `CONTENT_POLICY.md`
- Additional disclaimer: `DISCLAIMER.md`
- Non-compliance will result in account deletion and law enforcement reports.
