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
