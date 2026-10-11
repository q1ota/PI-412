# Рекомендательная система фильмов

Олег, группа 11-412, КФУ. Блок заданий 1, часть 3 RecSys.
Это отдельная часть командной работы; пункт 3 старого README относится к изображениям.

Реализованы два классических подхода — content-based и item-based collaborative filtering —
и две эвристики: популярность со сглаживанием оценок и та же популярность внутри жанра.
Реальные результаты находятся в `artifacts/recsys/`, разбор частей II–III — в
[отчёте](recsys-report.md), сценарий показа и устная защита — в [памятке](recsys-defense.md).

## Установка и данные

Рекомендуется Python 3.11 или 3.12. Все команды ниже выполняются из корня репозитория.
Для самостоятельного запуска RecSys достаточно NumPy и FastAPI; PyTorch и веса моделей
Hugging Face не нужны.

На этом компьютере окружения уже подготовлены; для демонстрации:

```bash
cd /Users/melnikoleg/Project/PI-412
source .venv-recsys/bin/activate
python -m recsys demo --user-id 1 --method item_cf --k 5
```

Для установки на новой машине с Python 3.11:

```bash
python3.11 -m venv .venv-recsys
source .venv-recsys/bin/activate
python -m pip install -r requirements-recsys.txt
python -m recsys download
```

В Windows создание: `py -3.11 -m venv .venv-recsys`, активация:
`.venv-recsys\Scripts\activate`. Вместо Python 3.11 можно использовать 3.12.
Полное окружение командного проекта на этом компьютере находится в `.venv`.

Источник данных выбран самостоятельно: в присланном условии конкретный набор не задан.
[MovieLens 100K от GroupLens](https://grouplens.org/datasets/movielens/100k/) содержит
оценки 1–5, время оценок, названия и жанры фильмов. Команда `download` скачивает
официальный архив по HTTPS, сверяет SHA-256 и сохраняет необходимые файлы в `data/ml-100k/`.
Точный SHA-256 указан в `recsys/data.py` и результатах эксперимента.
Исходный `README` набора с условиями использования сохраняется рядом с данными.
Демографический файл `u.user` не извлекается и не используется.

Для учебного исследования используется разрешённый исследовательский сценарий.
Лицензия исходного набора запрещает перераспространение без разрешения и требует
отдельного разрешения для коммерческого использования. Поэтому данные исключены
из Git; каждому участнику следует скачать их с GroupLens.
Подробности: [официальный README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt).

## Эксперимент и демонстрация

```bash
python -m recsys evaluate
python -m recsys demo --user-id 1 --method content --k 5
python -m recsys demo --user-id 1 --method item_cf --k 5
python -m recsys demo --user-id 1 --method popular --k 5
python -m recsys demo --user-id 1 --method genre_popular --genre Sci-Fi --k 5
python -m recsys demo --user-id 9999 --method item_cf --k 5
python -m recsys audit
```

`evaluate` сохраняет метрики, протокол и примеры в `artifacts/recsys/results.json`,
результаты по каждому пользователю — в `per_user.csv`, а краткую таблицу — в `metrics.csv`.
Для повторного запуска без перезаписи приложенных результатов:

```bash
python -m recsys evaluate --output /tmp/recsys-check
python -m recsys audit --output /tmp/recsys-check/shilling.json
```

Последняя команда — контролируемое исследование накрутки рейтингов в памяти;
исходные файлы MovieLens остаются неизменными.

## Локальное развёртывание HTTP-сервиса

```bash
python -m uvicorn recsys.api:app --host 127.0.0.1 --port 8001
```

Открыть [Swagger](http://127.0.0.1:8001/docs) и вызвать `GET /recsys/recommend`.
В другом терминале:

```bash
curl 'http://127.0.0.1:8001/recsys/recommend?user_id=1&method=item_cf&k=5'
curl 'http://127.0.0.1:8001/recsys/recommend?user_id=9999&method=content&k=5'
curl 'http://127.0.0.1:8001/recsys/recommend?user_id=1&method=genre_popular&genre=Sci-Fi&k=5'
```

Поддерживаются `content`, `item_cf`, `popular`, `genre_popular`; `k` — от 1 до 50.
`genre` применяется только к `genre_popular`; без него жанр берётся из положительной
истории. Ответ содержит фильм, жанры, балл и объяснение. Балл — внутренний показатель
для сортировки, он не является вероятностью, предсказанной оценкой или величиной,
которую можно сравнивать между разными методами.

Модель обучается один раз на первом запросе, затем кэшируется в памяти процесса.
Без скачанных данных API возвращает 503 с командой загрузки. Неизвестный положительный
ID пользователя даёт сценарий cold start, а не ошибку. Запрос не скачивает данные.
`/health` проверяет только работоспособность процесса, готовность модели проверяется
реальным запросом рекомендаций. При обновлении данных сервис требуется перезапустить.

## Совместимость с командным API и тесты

Роутер также подключён к `main:app`; старые пути и их реализация сохранены.
Для полного проекта установить исходные зависимости команды:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

На Intel macOS с Python 3.11/3.12 использовать проверенные совместимые версии:

```bash
python -m pip install -r requirements.txt -c constraints-macos-intel.txt
python -m pytest -q
```

Это нужно из-за ограничения доступных Intel macOS wheels PyTorch и несовместимости
его старой версии с NumPy 2 и свежими Transformers. Полный список реально установленных
пакетов полного окружения сохранён в `artifacts/recsys/environment.txt`,
лёгкого окружения — в `environment-recsys.txt`.
Прекращение сборок Intel macOS после серии 2.2 подтверждено
[объявлением команды PyTorch](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).

При установке только лёгких зависимостей можно выполнить автономные проверки:

```bash
python -m pytest tests/test_recsys.py -q -k 'not router_is_integrated'
```

Одна проверка подключения к `main.py` требует полного окружения. Старые API-тесты
подменяют нейросетевые модели. Тесты RecSys проверяют настоящие расчёты на малом наборе;
отдельно проведены эксперимент на полном MovieLens и HTTP-проверка настоящей модели.
GitHub Actions дополнен отдельным заданием с загрузкой данных и реальным экспериментом.
Запуск Actions в GitHub произойдёт после будущего push; локальный запуск не заменяет его.

## Состав вклада

`recsys/` — данные, модели, метрики, исследование накруток, CLI и FastAPI-роутер.
`tests/test_recsys.py` — проверки своей части. `docs/recsys*.md` — условие, инструкции,
отчёт и защита. `artifacts/recsys/` — фактические результаты и сведения о проверках.
В `main.py` добавлены только импорт и подключение роутера. README, REPORT, `.gitignore`
и CI дополнены для интеграции. Старый ноутбук не изменён.
