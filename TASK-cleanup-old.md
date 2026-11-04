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
