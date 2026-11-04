# Задача: убрать старое на `stt` и `ai-tools` (exec-api, stt-api, старый тестовый ai-trns)

Поставлено: 2026-11-04. Разрешение пользователя: «на stt можно удалить старое, пока оставить mdb-lora для абзацев;
на ai-tools можно убрать старое использование exec (включая nginx endpoints)».
Связано: `TASK-speakers-paragraphs.md` (diarize и абзацы должны жить в ai-trns **до** удаления stt-api).

## Что оставить
- `stt`: **`/home/mdb-lora`** (conda env `mdb-lora`, веса `adapters/paragraphing/outputs/paragraphing`) — модель абзацев.
  Образцы голоса `/home/stt-api/diarize/speakers/` — **сначала скопировать** в `AI_TRNS_SPEAKERS_DIR` на `ai-tools`.
- `ai-tools`: `ai-trns-api`, `ai-trns-worker`, `ai-trns-live-worker`, `ollama` (bge-m3 — проверить, кто им пользуется, не трогать без ответа).
- MQTT-брокер — не трогаем (может обслуживать не только ai-tools); убираем только его использование в клиенте.

## Что убрать
| Где | Что | Сейчас (2026-11-04) |
|---|---|---|
| `stt` | `stt-api.service` (uvicorn :8765, 4,6 ГБ GPU), `/home/stt-api` | живой: `/diarize/*`, `/tadiff`, `/docdiff`, `/review`, `/stt/*`-прокси |
| `stt` | `ai-trns.service` — старый тестовый `ai_trns.server` (:8766, 2,2 ГБ GPU), старая `/home/ai-trns` | живой |
| `stt` | `/home/stt-api-client*`, `/home/text_audio_align`, `/home/wav2vec2-asr-pipeline` | проверить, не нужны ли (`text_audio_align` — выравнивание текста по аудио, может понадобиться прототипу фрагментов) |
| `ai-tools` | `exec-api.service` (порт 8081, `/home/amnonbb/trns`), `whisper_fullfile_cli.py` в ai-trns | живой |
| Angie `ngx.kab.info` | locations для exec-api (`/rest/`…) и stt-api (`https://ai-tools.kab.info/stt-api/…`) | найти в конфиге |
| клиент `../ai-tools` | `src/api/exec.js`, `src/api/mqtt.js`, `STT_URL`/`REACT_APP_STT_URL`, `EXEC_API`, `MQTT_URL` | используются в `pages/Transcription.jsx`, `pages/Speakers.jsx`, `api/transcribe.ts`, `api/UserManager.js` |

## Условия до удаления (иначе сломаем живые страницы)
1. **Клиент ai-tools переведён на ai-trns** во всех страницах, задеплоен с `master`:
   - `Transcription.jsx` → `POST /jobs` (уже есть в ветке `ai-trns-rest` — проверить, что влито и задеплоено);
   - `Speakers.jsx` → `/diarize` ai-trns (задачи 202 + опрос);
   - убраны `exec.js`, `mqtt.js`, `STT_URL`, `EXEC_API`, `MQTT_URL` из кода, `.env` и Actions variables.
2. **Diarize работает в ai-trns на `ai-tools`**: воркер с движком `diarize` (`requirements-diarize.txt`, torch под драйвер 535,
   `HF_TOKEN`, образцы голоса), проверено на уроке — совпадение с stt-api (как в TODO: слово в слово).
3. **Трафика нет:** логи Angie — после деплоя клиента нет запросов к locations exec-api и stt-api, кроме health-check
   (посмотреть, кто ходил за последние 14 дней, — если кто-то кроме ai-tools, остановиться и сообщить). Записать цифры сюда.
4. Данные review уже перенесены (TODO, 2026-10-20) — сверить, что после 10-20 в `stt:/home/stt-api/data/review` ничего нового.

## Порядок (уточнено с пользователем 2026-11-04: ждать 3–7 дней не нужно, удалять сразу после проверки)
**Этап A — exec-api на `ai-tools` (не ждёт diarize):**
1. Страница תמלול (`Transcription.jsx`) → ai-trns `/jobs`: ветка клиента `ai-trns-rest` уже это делает — влить, задеплоить, проверить
   под пользователем (урок → SRT). exec-api последний раз запускался 2026-10-21 (`/home/amnonbb/trns/output.srt`);
   он грузит faster-whisper на ту же RTX 2080 — делит память с ai-trns.
2. Бэкап (`/home/amnonbb/trns` конфиги, unit, Angie) → удалить `exec-api.service`, location exec-api в Angie, `exec.js`/`mqtt.js`/`EXEC_API`/`MQTT_URL` в клиенте
   (если их не использует ещё какая-то страница — проверить), `whisper_fullfile_cli.py` в ai-trns.

**Этап B — stt-api и старый ai-trns на `stt` (после diarize в ai-trns):**
1. Бэкап: `/home/stt-api/data`, `/home/stt-api/diarize/speakers`, unit-файлы, конфиги Angie → архив на `ai-tools`.
2. Diarize-воркер на `ai-tools` + `Speakers.jsx` на ai-trns, проверено.
3. Удалить `stt-api.service`, `ai-trns.service` (старый тестовый) и их каталоги, location stt-api в Angie, `STT_URL` в клиенте.
4. `text_audio_align`, `wav2vec2-asr-pipeline`, `stt-api-client*` — **не удалять**, перечислить пользователю, что в них, и спросить.
5. Упоминания exec-api и stt-api в README, ARCHITECTURE, TODO — убрать/обновить.
5. На `stt` после удаления: свободно ~6,9 ГБ GPU → записать, что теперь там: mdb-lora (абзацы) и, при желании,
   воркер ai-trns (TODO «Воркер на `stt`»: нужна связь `stt → API`).

## Откат
Всё обратимо из бэкапа. Если после удаления что-то сломалось — восстановить из архива и вернуть location.

## Готово, когда
- Старые сервисы удалены, бэкап есть, клиент без exec/MQTT/stt-api, все страницы ai-tools работают через ai-trns.
- TODO.md (раздел «Старая схема exec-api» — удалить), README, ARCHITECTURE обновлены.
- Здесь записано: что удалено, где бэкап, сколько GPU освободилось.

## Ход работ

### Логи Angie (2026-11-04, `ngx.kab.info:/var/log/angie/access.log*`, 20.10–04.11; хоста в формате лога нет — location `/rest/`, `/stt-api/`, `/whisper/` есть только в `conf.d/ai-tools.conf`)
| location | Запросов | Кто |
|---|---|---|
| `/rest/` (exec-api) | 52 | реальные — только страница `https://ai-tools.kab.info/transcription` (20.10, 21.10 — пользователь; **23.10 — другой пользователь, 176.229.114.205, загрузка файла**); остальное — сканеры (404) |
| `/stt-api/` | 65 | только ai-tools (`/review`, `/stt`) и `localhost:3000/3123` (разработка); последний — 20.10 |
| `/whisper/` | 21 | только stt-api с `stt` (84.108.114.101 = NAT `stt`); последний — 20.10 |
Чужих потребителей нет.

### Этап A — сделано 2026-11-04
- [x] Страница Transcription (`Transcription.jsx`, английская; תמלול — это `Stt.tsx`, она уже была на `/jobs`) переписана на ai-trns
      `POST /jobs`: файл или ссылка, язык he/en/ru/auto, форматы srt/txt, прогресс задачи, скачивание `/jobs/{id}/result`.
      ai-tools PR #4 (`feature/ai-trns-only` → `master`, в него вошли `ai-trns-rest` и `feature/transcript-compare`),
      задеплоено Actions с `master` (run 41795365419).
- [x] Сервер: воркер с `AI_TRNS_STT_URL` отправляет в whisper_trainer только иврит по URL; загрузки и en/ru/auto — на свои
      модели. Модели `en` (whisper-large-v3-ct2) и `ru` (antony66 whisper-large-v3-russian ct2) скопированы из
      `/home/amnonbb/trns/models` в `/home/ai-trns/models`. `AI_TRNS_TRANSCRIBE_UPLOADS=1`. Исправлен профиль `stable`
      (`log_prob_threshold`). Задеплоено, бэкап `/home/ai-trns/app.bak.20261104094214`, env `ai-trns-api.env.bak.20261104094214`.
- [x] Проверка (через Angie, ключ сервиса; войти под пользователем Keycloak агент не может — **проверить под пользователем**):
      загрузка he 2 мин → done за 7 с, SRT с RTL-метками (`fix_rtl`, как делал exec-api); en 1 мин → 7 с; auto → 7 с; CORS preflight 200.
- [x] Бэкап: `ai-tools:/root/backup-old-20261104/exec-api/` — unit, `trns-config-scripts-logs.tgz` (всё из `/home/amnonbb/trns`
      кроме `tmp/` 15 ГБ загрузок, `models/` 7,4 ГБ, `hosting/` 2,9 ГБ и mp3; их списки — `*-listing.txt`).
      Angie: `ngx.kab.info:/root/ai-tools.conf.bak.20261104095453`.
- [x] Удалено: `exec-api.service` (остановлен, disable, unit удалён; порт 8081 свободен); location `/rest/` в Angie (`angie -t`, reload);
      в клиенте `exec.js`, `mqtt.js`, `MQTT_URL`/`EXEC_API`/`AI_BACKEND`/`WEB_URL` (код, `.env`, `.env.example`, workflow,
      Actions variables — значения были: `msg.kab.info`, `https://ai-tools.kab.info/rest`, `…/rest/upload/file`, `https://demux.kab.sh`),
      пакеты `mqtt`, `rc-upload`; `whisper_fullfile_cli.py` в ai-trns.
- **Каталог `/home/amnonbb/trns` (25 ГБ, домашний каталог другого человека) не удалён** — решить пользователю.

### Diarize в ai-trns на `ai-tools` — сделано 2026-11-04
- [x] Отдельный воркер `ai-trns-diarize-worker` (unit и env-пример в `scripts/`), свой venv `/home/ai-trns/env-diarize`
      (torch 2.14.1+cu126 работает с драйвером 535; pyannote.audio 4.0.7, torchaudio 2.11), чтобы torch не попал в env live/transcribe.
      `HF_TOKEN` взят из `stt:/home/stt-api/.env` (в `ai-trns-diarize-worker.env`, права 600). Образцы `Rav` → `/home/ai-trns/models/speakers`.
- [x] Сверка со stt-api: урок `cdn.kabbalahmedia.info/3RhEpAgW` (78 мин, 855 реплик) — **5885/5885 слов с тем же спикером и
      тем же началом** (статистика одинаковая: 18 спикеров, `Rav` найден). ai-trns на RTX 2080 — 172 с, stt-api на RTX 3070 — 122 с.
- [x] Память GPU `ai-tools`: live 1,8 ГБ, transcribe 2,1 ГБ (модель грузится только для загрузок/других языков/фолбэка),
      diarize 4,1 ГБ. Все трое разом — 7,94 из 8 ГБ → добавлен порог `AI_TRNS_MIN_FREE_GPU_MB`: воркер без загруженной модели
      не берёт задачу, пока свободно меньше порога. diarize — 6200 (во время трансляции не стартует, ждёт), transcribe — 4000;
      `IDLE_UNLOAD_S`: diarize 30, transcribe 60, live 60. Проверено: diarize во время live стоит в очереди, после — выполняется.
- [x] Клиент: `api/stt.js` → `api/tools.js` (Speakers, Doc diff, Text diff — задачи ai-trns); страницы по-прежнему выключены в меню.
      ai-tools PR #5 → `master`, задеплоено (run 41798758479). Из кода, `.env.example`, workflow и Actions variables убраны
      `REACT_APP_STT_API` (`https://ai-tools.kab.info/stt-api`) и `REACT_APP_STT_URL` (`…/stt-api/stt/transcribe`).
- **Найдено попутно:** сервис whisper_trainer `10.66.1.53:8765` 2026-11-04 отказывает в соединении (хост пингуется) → все URL-задачи
  на иврите падали. Воркер теперь при недоступном сервисе распознаёт своей моделью (`AI_TRNS_STT_FALLBACK=1` по умолчанию).

### Этап B — сделано 2026-11-04
- [x] Данные review: в `stt:/home/stt-api/data/review` после переноса (20.10 ~17:50) нового нет (последние файлы — 20.10 17:15,
      на `ai-tools` их более новые версии).
- [x] Бэкап на `ai-tools:/root/backup-old-20261104/stt/` (права 600): `stt-api-and-units.tgz` (вся `/home/stt-api` с `.env` и
      `data/`, unit'ы `stt-api.service` и `ai-trns.service`, `stt:/etc/nginx/conf.d/sttapi.conf`; 224 файла) и
      `old-ai-trns-home.tar` (вся старая `/home/ai-trns`, 4,8 ГБ, 15856 файлов). Angie: `ngx.kab.info:/root/ai-tools.conf.bak.20261104102339`.
- [x] Удалено на `stt`: `stt-api.service`, `ai-trns.service` (остановлены, disable, unit'ы удалены), каталоги `/home/stt-api`,
      `/home/ai-trns`. Порты 8765/8766 свободны. **GPU `stt`: было 6855 МиБ занято, стало 1 МиБ** (свободно ~8 ГБ).
- [x] Удалено в Angie: location `/stt-api/` (`angie -t`, reload). Сайт, `/ai-trns/health` — 200; задача по URL после удаления — done.
- **Не тронуто (нет в списке задачи, решить пользователю):**
  - Angie location `/whisper/` (прокси на whisper_trainer `10.66.1.53:8765` только для NAT `stt`, с ключом `X-STT-Key`) —
    им пользовался только stt-api (21 запрос, последний 20.10). Теперь не нужен.
  - `stt:/etc/nginx/conf.d/sttapi.conf` (`sttapi.kli.one`: статика `/home/stt-api-client` + `/api/` → :8765, теперь 502);
    трафика с 20.10 нет.
  - conda env `api` на `stt` (`/home/stt/miniconda3/envs/api`, окружение stt-api).
  - `ai-tools:/home/amnonbb/trns` (25 ГБ; exec-api), модели en/ru оттуда уже скопированы.
- Оставлены по задаче (что это):
  - `/home/text_audio_align` (78 ГБ: `data` 27 ГБ, `output` 52 ГБ) — репозиторий `Bnei-Baruch/text_audio_align`: выравнивание
    известного текста по длинной записи чтения (5–60 мин) → SRT по словам; может кормить `dataset_build` whisper_trainer.
    Последний коммит 2026-07-16. Пригодится прототипу «пункт источника → момент урока» и метке `קריין`.
  - `/home/wav2vec2-asr-pipeline` (12 ГБ, в основном `models/`) — репозиторий `Bnei-Baruch/wav2vec2-asr-pipeline`: своё ASR на
    wav2vec2 + API (`api/`, `api.log`), сравнение текстов; последний коммит 2026-07-07. Сервиса нет.
  - `/home/stt-api-client` (+ `.bak-20261006`) — собранная статика старого клиента stt-api (`index.html`, `assets/`), отдавалась
    nginx на `sttapi.kli.one`; без stt-api не работает.
- [x] Теперь на `stt`: `mdb-lora` (абзацы) и сервис `ai-trns-paragraphs` на :8766 (4,3 ГБ при работе, выгружается в простое).
- Образцы голоса Рава на `ai-tools`: 2 из stt-api + 1 новый (урок 30.11.24) — см. `TASK-speakers-paragraphs.md`.
