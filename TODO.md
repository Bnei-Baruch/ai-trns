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

### CLI для exec-api (совместимость)
- [x] Те же аргументы и файлы (`output.*`, `report_python.log`). `--impl` принимается и игнорируется.
- [x] `--out-dir`, `--report-file`, `--models-dir`, `--nginx-dir`, `--json`, `--profile`, `--batch-size`.
- [x] Финальный статус `[✓] DONE` / `[✗] ERROR: …`. Реальный прогресс по сегментам. `output.wav` удаляется при
      ошибке и при SIGTERM.

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
- [ ] Клиент ai-tools: страница транскрипции через API (`POST /jobs` с `kc.token`, опрос статуса, скачивание)
      вместо exec-api + MQTT.
- [ ] URL-задачи с `cdn.kabbalahmedia.info`: прогнать настоящую ссылку.
- [ ] Влить `rest-api` в master после переключения клиентов. Потом убрать `whisper_fullfile_cli.py` и exec-api.

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
- [ ] Старый тестовый сервис `ai-trns` на `stt` (держит GPU) — остановить или обновить до воркера.
- [x] stt-api снова запущен 2026-10-20 (по просьбе пользователя): GPU `stt` занят на 6.9 из 8 ГБ вместе со старым тестовым `ai-trns`.

### Из stt-api
- [x] `/review/*` перенесён (`ai_trns/api/review.py`, ветка `review`): урок = задача ai-trns, рецензент из токена.
      ai-tools (ветка `ai-trns-rest`): Review и תמלול ходят в ai-trns с токеном Keycloak, транскрипции в браузере нет.
- [x] `/stt/*`-прокси не нужен: תמלול вызывает `/jobs` напрямую.
- [ ] Перенос данных: `stt:/home/stt-api/data/review` → `ai-tools:/home/ai-trns/review` (права `ai-trns`),
      задать `AI_TRNS_REVIEW_DIR` в `ai-trns-api.env`. Уроки `pending` без задачи станут `failed` — добавить заново.
- [ ] ai-tools: `REACT_APP_AI_TRNS_API` в `.env` и в Actions variables; убрать `REACT_APP_STT_URL`.
- [ ] Проверить вживую: добавить урок в Review, дождаться `ready`, сохранить результат, открыть отчёт.
- [x] diarize / tadiff / docdiff перенесены (ветка `review`): запрос создаёт задачу, результат — `/jobs/{id}/result`.
      tadiff/docdiff — обычное распознавание + сравнение на API; diarize — движок воркера `diarize`.
      Движок проверен на `stt` против stt-api: урок 1:45 ч, вывод совпал слово в слово, 159 с на RTX 3070.
- [ ] Развернуть diarize: на GPU-хосте `requirements-diarize.txt` (на `ai-tools` драйвер 535 → torch под CUDA 12),
      `HF_TOKEN`, образцы голоса `stt:/home/stt-api/diarize/speakers/Rav` → `AI_TRNS_SPEAKERS_DIR`,
      `AI_TRNS_WORKER_ENGINES=transcribe,diarize`. Хватит ли 8 ГБ рядом с exec-api и ollama.
- [ ] ai-tools: если включать эти страницы — перейти на задачи (202 + опрос), поля запросов те же.

### Live: потоковая транскрипция
- [ ] Потоковая транскрипция трансляций (текст с задержкой в секунды). План, вопросы и шаги —
      [TASK-live-transcription.md](TASK-live-transcription.md). Источник — WebRTC через Janus, звук
      на воркер через `rtp_forward`. Сначала узнать: хост и версия Janus, плагин и комната, `admin_key`,
      проходит ли UDP Janus → GPU-хост; кто смотрит результат, задержка, GPU.

### Мелочи
- [ ] `/health` показывает воркеры с PID в id до 5 мин после рестарта — задать `AI_TRNS_WORKER_ID` везде.
- [ ] Предзагрузка следующей задачи в воркере (скачать и конвертировать, пока GPU занят) — для коротких файлов.
- [ ] Приоритеты в очереди (короткие файлы, review ниже), если понадобится.

## Старая схема exec-api (для справки)

exec-api запускает CLI как сервис из своего конфига, например
`{"ID": "python", "Name": "python3", "Args": ["whisper_fullfile_cli.py", "--input", "...", "--srt", "--lang", "he", "--fix-rtl"]}`.
Процесс наследует cwd exec-api, статус читается через `/:ep/report/python` → `report_python.log`.

Открытые проблемы exec-api (уйдут вместе с ним):
- **Watchdog** (`api/exec.go:28`) раз в 30 с перезапускает завершённый сервис без PID, exit code не проверяет.
- **REST exec-api** открыт: GET для start/stop, CORS `*`, авторизация выключена (`authentication.enable=false`).
