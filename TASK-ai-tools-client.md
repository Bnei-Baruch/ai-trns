# Задача: довести сервер ai-trns до того, что нужно клиенту ai-tools

От клиента ai-tools (`../ai-tools`, ветка `feature/transcript-compare`), 2026-10-20.

## Контекст
Клиент ai-tools теперь ходит в ai-trns напрямую, с токеном Keycloak пользователя (`Authorization: Bearer`):
- **תמלול** (`/stt`) и новая страница **השוואת תמלול** (`/compare`): `POST /jobs` (`url`, `lang=he`, `format=srt`) →
  опрос `GET /jobs/{id}` → `GET /jobs/{id}/result?format=srt`.
  Ссылка всегда на `https://cdn.kabbalahmedia.info/<file id>`.
- **בדיקת תמלול** (`/review`): `/review/*` (набор `main`), сохранение результата — `PUT`.
- Прод: клиент на `https://ai-tools.kab.info`, API — `https://ai-tools.kab.info/ai-trns`.
  Локальная разработка: клиент на `http://localhost:3000` обращается к тому же API напрямую (CORS).

## Что не так сейчас (проверено снаружи, 2026-10-20)
1. **На сервере старая версия.** В `https://ai-tools.kab.info/ai-trns/openapi.json` только `/health`, `/jobs*`,
   `/render`, `/worker/*`. Нет `/review/*`, `/diarize/*`, `/tadiff/*`, `/docdiff/*` — всё это лежит в рабочем
   дереве незакоммиченным. Из-за этого не работает בדיקת תמלול.
2. **`POST /jobs` с `url` → 422** (пользователь, страница השוואת תמלול, ссылка на `cdn.kabbalahmedia.info`).
   Поля запроса совпадают со схемой, значит это проверка URL в `create_job`: скорее всего на сервере пустая
   `AI_TRNS_ALLOWED_URL_HOSTS` («URL input is disabled») или в ней нет `cdn.kabbalahmedia.info`.
   В `scripts/ai-trns-api.env.example` значение правильное, на сервер оно, видимо, не попало.
3. **CORS без `PUT`.** Origin `http://localhost:3000` уже разрешён (пользователь добавил), но preflight отвечает
   `access-control-allow-methods: GET, POST, DELETE`. В текущем коде `PUT` есть — придёт с деплоем п. 1.

## Что сделать
1. Довести и закоммитить текущую работу (review / diarize / tadiff / docdiff), тесты зелёные.
2. Задеплоить на `ai-tools` (10.66.24.55) как в `docs/ARCHITECTURE.md` (rsync в `/home/ai-trns/app`, рестарт
   `ai-trns-api` и `ai-trns-worker`). exec-api и ollama не трогать.
3. В env API на сервере проверить/выставить:
   ```
   AI_TRNS_ALLOWED_URL_HOSTS=cdn.kabbalahmedia.info
   AI_TRNS_CORS_ORIGINS=https://ai-tools.kab.info,http://localhost:3000
   AI_TRNS_REVIEW_DIR=/home/ai-trns/review
   ```
   Если в stt-api (`stt`, `data/review`) остались наборы/результаты проверки — перенести в `AI_TRNS_REVIEW_DIR`
   (так сказано в env-примере).
4. Обновить `TODO.md` (раздел «Запуск для пользователей»).

## Проверка (готово, когда всё это проходит)
- `openapi.json` содержит `/review/sets/{set_id}`, `/review/sets/{set_id}/segments`,
  `/review/sets/{set_id}/reviewers/{reviewer}/segments/{segment_id}` и т.д.
- Preflight с `Origin: http://localhost:3000`, `Access-Control-Request-Method: PUT` → 200, `PUT` в allow-methods.
- `POST /jobs` (ключ сервиса) с `url=https://cdn.kabbalahmedia.info/K8X56H7k`, `lang=he`, `format=srt` → 202,
  задача доходит до `done`, `GET /jobs/{id}/result?format=srt` отдаёт SRT.
- `GET /review/sets/main` с ключом → 200.

## Не делать
- Не делать `/tadiff`, `/docdiff`, `/diarize` синхронными ради старого клиента: страницы Speakers / Doc diff /
  Text diff в ai-tools сейчас выключены и будут переведены на задачи (202 + опрос) на стороне клиента.
- `https://ai-tools.kab.info/ai_trns` (с подчёркиванием) отдаёт `index.html` клиента — это нормально,
  правильный адрес `/ai-trns`, алиас не нужен.

## Отчёт
Когда закончишь — напиши сессии `ai-tools-8f` (клиент): что задеплоено (коммит), какие env выставлены, результаты
проверок выше.
