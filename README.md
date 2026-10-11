# Мультимодальный анализ данных с использованием Hugging Face Transformers

Практический проект, демонстрирующий решение четырех базовых задач обработки данных различных модальностей (текст, аудио, изображения, видео) с использованием современных предобученных моделей глубокого обучения.

## 🛠 Используемые технологии
* **Python 3.13**
* **PyTorch** (фреймворк глубокого обучения)
* **Transformers (Hugging Face)** (библиотека для работы с pretrained-моделями)
* **Datasets** (загрузка демонстрационных датасетов)
* **Torchvision / OpenCV / Librosa** (обработка изображений, видео и аудио)

---

## 📌 Содержание проекта (Задачи)

1. **Анализ тональности текста (Sentiment Analysis)**
   * Используемая модель: `distilbert-base-uncased-finetuned-sst-2-english`.
   * Анализ эмоциональной окраски и скрытого контекста текстовых сообщений.
2. **Распознавание речи (Automatic Speech Recognition - ASR)**
   * Используемая модель: `openai/whisper-tiny`.
   * Транскрипция аудиофрагментов в текстовый формат.
3. **Классификация изображений**
   * Используемая модель: `ResNet-50` (предобучена на ImageNet).
   * Определение объектов на фотографиях с выдачей топ-5 вероятных классов.
4. **Анализ видеофрагментов (кадровый анализ)**
   * Используемая модель: `google/vit-base-patch16-224` (Vision Transformer).
   * Извлечение и классификация ключевых кадров видеоряда.

---

## 🚀 Быстрый старт

Проект полностью адаптирован для запуска в среде **Google Colab**. Вы можете открыть файл ноутбука (`.ipynb`) непосредственно в облачной среде и выполнить все ячейки последовательно.

## API на FastAPI

API предоставляет доступ к моделям проекта для анализа текста, распознавания речи, классификации изображений и анализа кадров видео.

### Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

После запуска откройте `http://127.0.0.1:8000/docs` для интерактивной документации Swagger. Веса моделей загружаются из Hugging Face или Torchvision при первом запросе соответствующего типа; для первой загрузки нужен доступ к интернету.

### Эндпоинты

| Метод и путь | Вход | Результат |
| --- | --- | --- |
| `GET /health` | — | Состояние API |
| `POST /text/sentiment` | JSON: `{"text": "I liked the result"}` | Метка и уверенность DistilBERT |
| `POST /audio/transcribe` | Файл `file` (WAV, FLAC или OGG) | Распознанный текст Whisper |
| `POST /image/classify` | Файл `file`, параметр `top_k` от 1 до 5 | Классы изображения ResNet-50 |
| `POST /video/analyze` | Видео `file`, интервал выборки `interval_seconds` | Классы выбранных кадров ViT |

Пример запроса для текста:

```bash
curl -X POST http://127.0.0.1:8000/text/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"I liked the result"}'
```

Для загрузки файлов используйте форму эндпоинта в `/docs`. Максимальный размер загружаемого файла — 50 МБ; для видео анализируется не более 30 кадров.

### Тесты и CI

Тесты API запускаются командой `python -m pytest -q`. GitHub Actions автоматически запускает их при каждом push и pull request.

## Блок заданий 1 — часть 3 RecSys

Вклад Олега, группа 11-412: рекомендации фильмов на MovieLens 100K.
Два классических подхода (content-based и item-based collaborative filtering)
и две эвристики (популярность и популярность внутри жанра), реальный эксперимент,
локальное HTTP-развёртывание и разбор ограничений, мониторинга и бизнес-метрик.
Эта часть отдельна от классификации изображений в пункте 3 выше.

```bash
python3.11 -m venv .venv-recsys
source .venv-recsys/bin/activate
python -m pip install -r requirements-recsys.txt
python -m recsys download
python -m recsys evaluate
python -m recsys demo --user-id 1 --method item_cf --k 5
python -m uvicorn recsys.api:app --host 127.0.0.1 --port 8001
```

HTTP: `GET /recsys/recommend?user_id=1&method=item_cf&k=5`.
Роутер подключён и к существующему `main:app`.

Подробности: [запуск и данные](docs/recsys.md),
[отчёт по частям I–III](docs/recsys-report.md),
[демонстрация и защита](docs/recsys-defense.md),
[исходный текст условия](docs/recsys-assignment.md).
Результаты: [metrics.csv](artifacts/recsys/metrics.csv),
[полный эксперимент](artifacts/recsys/results.json).
Для полного проекта на Intel macOS предусмотрен `constraints-macos-intel.txt`;
способ установки приведён в инструкции RecSys.
