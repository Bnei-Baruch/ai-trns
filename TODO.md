# TODO

Как всё устроено и где что лежит: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Сделано (ветка `rest-api`, 2026-10-20)

Коммиты: `875f9ac` → `fe509ee` → `e73d20c` → `0afb138` (не запушено, в master не влито).

### Структура и ядро
- [x] Код разложен в пакет `ai_trns/`. Глобальное состояние убрано: класс `Transcriber`, параметры передаются явно.
- [x] Документ со словами (`words.py`) — граница между GPU-распознаванием и всем остальным. Склейка слов с
      гершаимом и интерполяция перенесены из whisper_trainer.
- [x] SRT только по алгоритму whisper_trainer (`cues.py`, перенесён 1:1 с @29645c8), с их калиброванной поправкой
      таймкодов. `ToSubs`, «standard» SRT и pysrt удалены. `fix_rtl` оставлен для иврита.
- [x] Модель иврита — `orendar/whisper-large-v3-turbo-hebrew-kabbalah-ct2` (те же веса, что в текущем проде).
      Профиль `served`: пакетные VAD-окна, greedy. Настройки модели в `ai_trns_model.json`.
- [x] Звук всегда декодирует ffmpeg (`-protocol_whitelist`). PyAV не нужен, он несовместим с faster-whisper 1.2.1.

### API и воркеры
- [x] API-сервер (`ai_trns/api`, FastAPI) без GPU: задачи в SQLite, файл или URL (allowlist хостов), форматы
      `srt,txt,json`, прогресс, отмена, TTL, лимит загрузки. Любой формат можно перерисовать из слов
      (`/jobs/{id}/result?format=…`, `POST /render`).
- [x] GPU-воркер (`ai_trns/worker`) — отдельный процесс, сам забирает задачи (pull) по протоколу
      `ai_trns/protocol.py`. Аренда продлевается heartbeat'ом каждые 5 с, при потере воркера задача возвращается
      в очередь, после 3 попыток — `error`. Отмена доходит через heartbeat. Модель выгружается после 5 мин простоя.
- [x] Авторизация: пользователи — токены Keycloak (realm `main`, клиент `ai-tools`, роль `bb_user`), сервисы —
      именованные ключи, воркеры — свой ключ. Владелец у задачи, пользователь видит только свои. `GET /jobs`.
      CORS для `https://ai-tools.kab.info`.
- [x] Тесты (38): реплики (перенесены из whisper_trainer), слова, API, протокол воркера, аренда, отмена,
      авторизация, настоящий воркер против настоящего API-сервера.

### Развёрнуто
- [x] `ai-tools` (10.66.24.55): systemd `ai-trns-api` (порт 8766) и `ai-trns-worker` (RTX 2080) от пользователя
      `ai-trns` в `/home/ai-trns`. Проверено: урок 41 мин → SRT за 30 с. exec-api и ollama не тронуты.
- [x] `stt` (10.65.4.100): тестовая установка `/home/ai-trns` и замеры на RTX 3070 (105 мин за ~50 с, 3.4 ГБ).

### Анализ
- [x] whisper_trainer: что в нём относится к использованию модели. Их `transcription_service.py` — это и есть
      upstream 10.66.1.53:8765, на который проксирует stt-api.
- [x] Сравнение SRT whisper_trainer / stt-api (diarize) / старый ai-trns. Сравнение с их продом: разбивка
      совпадает, расхождение около 4% слов идёт от их декодера.
- [x] Сеть: `stt → ai-tools` закрыто, `ai-tools → stt` открыто.

## Дальше

### Запуск для пользователей
- [x] Angie на `ngx.kab.info`: location `https://ai-tools.kab.info/ai-trns/` → `10.66.24.55:8766`
      (2026-10-20, бэкап `/root/ai-tools.conf.bak.20261020152425`). `/health` отвечает, `/jobs` без токена — 401.
- [x] Задеплоен `b297acc` (ветка `review`) в `ai-tools:/home/ai-trns/app` (2026-10-20): `/review`, `/diarize`, `/tadiff`,
      `/docdiff` в openapi; CORS `https://ai-tools.kab.info,http://localhost:3000` с `PUT`; `AI_TRNS_REVIEW_DIR=/home/ai-trns/review`;
      python-docx в env. Задача по URL `cdn.kabbalahmedia.info` → done → SRT. exec-api (`/home/amnonbb/trns`, master) не тронут.
- [x] Данные review из `stt:/home/stt-api/data/review` перенесены в `ai-tools:/home/ai-trns/review` (2026-10-20, без
      перезаписи: `main/KB74kOBz` оставлен новый, от whisper_trainer; бэкап `review.bak.*`).
- [ ] Проверить настоящий токен Keycloak пользователя (тестами покрыто, вживую не проверено).
- [ ] Решить, нужна ли роль администратора (`AI_TRNS_ADMIN_ROLE`).
- [x] Клиент ai-tools: страница Transcription через API (`POST /jobs` с токеном Keycloak, опрос, скачивание)
      вместо exec-api + MQTT (2026-11-04, ai-tools PR #4 → master, задеплоено). exec-api удалён — `TASK-cleanup-old.md`.
- [ ] URL-задачи с `cdn.kabbalahmedia.info`: прогнать настоящую ссылку.
- [ ] Влить ветки ai-trns (`review` → `cleanup-old`) в master. `whisper_fullfile_cli.py` и exec-api убраны (2026-11-04).

### whisper_trainer (полный прод-конвейер)
- [x] Временно: транскрипт из их сервиса `10.66.1.53:8765` (`ai_trns/stt_service.py`, `AI_TRNS_STT_URL` у воркера),
      SRT как есть; загрузка файлов для transcribe/tadiff/docdiff выключена (`AI_TRNS_TRANSCRIBE_UPLOADS=0`).
- [x] Задеплоено на ai-tools 2026-10-20 (рабочее дерево поверх `b297acc`, бэкап `app.bak.20261020143827` и env):
      URL-задача `cdn.kabbalahmedia.info/3RhEpAgW` → SRT байт в байт как у сервиса (858 реплик), загрузка файла → 422.
- [ ] Получить от Орена bundle v16 (`slices_soup_ctc_fw_v16`) — сообщение отправлено.
- [ ] Собрать их форк CTranslate2 под целевую машину (`tools/build_ctranslate2_wheel.sh`):
      `stt` — glibc 2.34; `ai-tools` — sm_75 и драйвер 535 (их torch 2.14 / CUDA 13 не пойдёт без нового драйвера).
- [ ] Движок воркера `whisper_trainer`: их `ResidentInferenceRuntime` из их окружения → наш `words.json`.
      Проверить, влезет ли в 8 ГБ (пачка 32 зафиксирована в bundle, языковой гейт).
- [ ] Взять из их сервиса: переиспользование результата по хешу URL; watchdog 900 с + 0.5 × длительность.

### Воркер на `stt`
- [ ] Связь `stt → API`: открыть порт `stt → ai-tools`, или через Angie, или push от API.
- [x] Старый тестовый `ai-trns` и stt-api на `stt` удалены (2026-11-04): GPU `stt` свободна (mdb-lora — абзацы).

### Из stt-api
- [x] `/review/*` перенесён (`ai_trns/api/review.py`, ветка `review`): урок = задача ai-trns, рецензент из токена.
      ai-tools (ветка `ai-trns-rest`): Review и תמלול ходят в ai-trns с токеном Keycloak, транскрипции в браузере нет.
- [x] `/stt/*`-прокси не нужен: תמלול вызывает `/jobs` напрямую.
- [ ] Перенос данных: `stt:/home/stt-api/data/review` → `ai-tools:/home/ai-trns/review` (права `ai-trns`),
      задать `AI_TRNS_REVIEW_DIR` в `ai-trns-api.env`. Уроки `pending` без задачи станут `failed` — добавить заново.
- [x] ai-tools: `REACT_APP_AI_TRNS_API` в `.env` и в Actions variables; `REACT_APP_STT_URL`, `REACT_APP_STT_API` убраны (2026-11-04).
- [ ] Проверить вживую: добавить урок в Review, дождаться `ready`, сохранить результат, открыть отчёт.
- [x] diarize / tadiff / docdiff перенесены (ветка `review`): запрос создаёт задачу, результат — `/jobs/{id}/result`.
      tadiff/docdiff — обычное распознавание + сравнение на API; diarize — движок воркера `diarize`.
      Движок проверен на `stt` против stt-api: урок 1:45 ч, вывод совпал слово в слово, 159 с на RTX 3070.
- [x] Diarize развёрнут на `ai-tools` (2026-11-04): `ai-trns-diarize-worker`, venv `env-diarize` (torch cu126), совпал со
      stt-api слово в слово на уроке 78 мин; порог памяти GPU `AI_TRNS_MIN_FREE_GPU_MB` — подробности в `TASK-cleanup-old.md`.
- [x] ai-tools: Speakers / Doc diff / Text diff — на задачах ai-trns (202 + опрос), страницы пока выключены в меню.
- [x] stt-api и старый тестовый ai-trns на `stt` удалены (2026-11-04), бэкап — `TASK-cleanup-old.md`.

### Live: потоковая транскрипция
- [ ] Потоковая транскрипция трансляций (текст с задержкой в секунды). План, вопросы и шаги —
      [TASK-live-transcription.md](TASK-live-transcription.md). Источник — WebRTC через Janus, звук
      на воркер через `rtp_forward`. Сначала узнать: хост и версия Janus, плагин и комната, `admin_key`,
      проходит ли UDP Janus → GPU-хост; кто смотрит результат, задержка, GPU.
- [x] Вариант 1 (Whisper + LocalAgreement): ядро и CLI `python -m ai_trns.live` (2026-10-21), тесты на фейковой модели.
- [x] Сессии live в API (`/live`, SSE), движок воркера `live`, отдельный `ai-trns-live-worker`, страница `תמלול חי`
      в ai-tools (2026-10-22). Тесты: 158 зелёных.
- [x] Задеплоено 2026-10-22: ai-trns на `ai-tools` (`ai-trns-live-worker`), ai-tools `41845fb` → `/live`.
- [x] Источник — HLS трансляции вместо WebRTC/Janus (2026-10-22): `https://edge3.uk.kab.tv/live/heb-medium/playlist.m3u8`,
      в ai-tools — `REACT_APP_LIVE_STREAM_URL` (.env и Actions variables), в API `AI_TRNS_ALLOWED_URL_HOSTS` += `.kab.tv`
      (точка в начале — любой поддомен). Проверено на трансляции: первый текст через ~3 с, проход 0.3 с,
      текст отстаёт от звука на ~4–5 с. Слова делятся по тексту сегментов, не по токенам (faster-whisper клеит слова
      на стыках сегментов).
- [x] Трансляция — одна задача на всех (`POST /live` с тем же потоком возвращает идущую), «עצור» останавливает только
      у последнего слушателя, без слушателей 60 с — задача заканчивается сама (2026-10-22).
- [x] Google Docs (2026-10-22), каждый способ — только если настроен:
      1) сервер на одном аккаунте (`AI_TRNS_GOOGLE_CREDENTIALS`: сервисный аккаунт или authorized_user,
      `AI_TRNS_GOOGLE_FOLDER`, `AI_TRNS_GOOGLE_DOC_ROLE`) создаёт документ при создании live-задачи и дописывает в него;
      2) браузер пишет в личный документ пользователя от его имени (ai-tools `REACT_APP_GOOGLE_CLIENT_ID`), список
      личных документов с владельцами виден всем слушателям. Правки — в самих документах, автора хранит Google.
- [ ] Настроить Google Cloud (`ai-tools-512410`) — по шагам со ссылками в [docs/GOOGLE-SETUP.md](docs/GOOGLE-SETUP.md):
  - [x] OAuth client (`REACT_APP_GOOGLE_CLIENT_ID`) и API key для Picker (`REACT_APP_GOOGLE_API_KEY`) прописаны в ai-tools.
  - [x] `privacy.html` и файл подтверждения Search Console (`googledec045341cd72249.html`) на сайте.
  - [ ] Включены Docs / Drive / Picker API; scope только `drive.file`; Branding заполнен; Audience — Publish app.
  - [ ] Окно выбора документа: «API key invalid» — проверить ограничения ключа (раздел 4).
  - [ ] Search Console → Verify `https://ai-tools.kab.info/`, через 24 ч — снова проверка бренда.
  - [ ] Запись с сервера: аккаунт (service account + папка на общем диске или authorized_user) → JSON на `ai-tools`,
        `AI_TRNS_GOOGLE_*` в `ai-trns-api.env`.
- [ ] Проверить страницу `/live` под пользователем (токен Keycloak, SSE через Angie, hls.js-плеер, Google Doc).

### Спикеры и абзацы (`TASK-speakers-paragraphs.md`)
- [x] Абзацы: сервис mdb-lora на `stt` (`ai-trns-paragraphs`), diarize-задача с `paragraphs=true`, `?format=paragraphs`.
- [x] Метки רב / קריין / תלמיד; чтец по «объявлению чтения», иначе по тексту источника (LIKUTIM).
- [x] `/compare`: «הצג דוברים ופסקאות». Замер на 3 уроках — в TASK-файле.
- [ ] Абзацы на ASR-тексте: exact F1 0,44 (на чистом тексте 0,816) — дообучить адаптер на «ASR-текст → абзацы редактора».
- [x] Замер «רב» на 10 уроках 2007–2025, 11 образцов (2026-11-04, TASK-файл): Рав узнаётся везде; с поправкой на
      непомеченное чтение 97,6–99,9% на 7 из 10; «hoarse» ничего не меняет; мультикластер (`AI_TRNS_DIARIZE_EXTRA_THRESHOLD`)
      бьёт по ученикам — выключен. Ниже цели: `172hAykR` (расщеплённый голос), лекция без объявления чтеца.
- [ ] Live: метки и абзацы (шаг 2) — не начато, план и препятствия в TASK-файле.

### Сервис whisper_trainer
- [ ] `10.66.1.53:8765` 2026-11-04 отказывает в соединении (хост жив). Пока он лежит, воркер распознаёт своей моделью
      (`AI_TRNS_STT_FALLBACK`), в документе — имя локальной модели. Сообщить владельцам whisper_trainer.

### Мелочи
- [ ] `/health` показывает воркеры с PID в id до 5 мин после рестарта — задать `AI_TRNS_WORKER_ID` везде.
- [ ] Предзагрузка следующей задачи в воркере (скачать и конвертировать, пока GPU занят) — для коротких файлов.
- [ ] Приоритеты в очереди (короткие файлы, review ниже), если понадобится.

