# Google Cloud: настройка записи live-текста в Google Docs

Страница `תמלול חי` (`https://<ai-tools>/live`) пишет текст трансляции в Google Docs двумя способами:

1. **Из браузера, от имени пользователя** — в его личный документ. Нужен OAuth client и API key (ниже, разделы 1–6).
2. **С сервера, на одном аккаунте** — ai-trns сам создаёт документ трансляции. Нужен аккаунт сервера (раздел 7).

Каждый способ включается, только если настроен. Проект: **`ai-tools-512410`**, номер проекта **`305720010110`**
(первая часть client ID).

Состояние на 2026-10-22: client ID и API key созданы и прописаны; файл подтверждения сайта выложен;
проверка бренда — ждёт подтверждения в Search Console + 24 ч. Запись с сервера не настроена.

## 1. Включить API

Каждая страница должна показывать «API Enabled»:

- Google Docs API — https://console.cloud.google.com/apis/library/docs.googleapis.com?project=ai-tools-512410
- Google Drive API — https://console.cloud.google.com/apis/library/drive.googleapis.com?project=ai-tools-512410
- Google Picker API — https://console.cloud.google.com/apis/library/picker.googleapis.com?project=ai-tools-512410

Проверить список: https://console.cloud.google.com/apis/dashboard?project=ai-tools-512410

## 2. Экран согласия (Google Auth Platform)

**Branding** — https://console.cloud.google.com/auth/branding?project=ai-tools-512410
(если не грузится: https://console.cloud.google.com/apis/credentials/consent?project=ai-tools-512410 → Edit app)

| Поле | Значение |
|---|---|
| App name | `AI-Tools` (это видят пользователи; иначе Google показывает домен «kab.info») |
| User support email | адрес поддержки (он же контакт в политике конфиденциальности) |
| Developer contact | e-mail для писем от Google |
| App logo | не загружать без нужды: логотип требует проверки бренда |
| Application home page | `https://<ai-tools>` |
| Privacy policy | `https://<ai-tools>/privacy.html` (лежит в `ai-tools/public/privacy.html`) |
| Authorised domains | `kab.info` |

**Data Access (scopes)** — https://console.cloud.google.com/auth/scopes?project=ai-tools-512410
Только `https://www.googleapis.com/auth/drive.file`. Это «нечувствительный» доступ: проверка приложения Google не
нужна. `documents` не добавлять — он чувствительный и требует проверки (видео, обоснование, недели ожидания).

**Audience** — https://console.cloud.google.com/auth/audience?project=ai-tools-512410
- External (пользователи с любыми Google-аккаунтами) → **Publish app** (In production).
- В режиме Testing работают только аккаунты из Test users.
- Internal — только если проект в Google Workspace и у всех рабочие аккаунты этой организации.

## 3. OAuth client

https://console.cloud.google.com/auth/clients?project=ai-tools-512410

- Тип **Web application**, клиент `305720010110-8q0k0gruc6dg8o1dniv5t0vc92hvdlre.apps.googleusercontent.com`.
- **Authorized JavaScript origins:** `https://<ai-tools>`, `http://localhost:3000`.
- Redirect URIs — пусто. Client secret не нужен.
- В ai-tools: `REACT_APP_GOOGLE_CLIENT_ID` (Actions variable + `.env`).

## 4. API key (окно выбора документа, Google Picker)

https://console.cloud.google.com/apis/credentials?project=ai-tools-512410 → ключ → Edit

- **Application restrictions:** Websites → `https://<ai-tools>/*`, `http://localhost:3000/*`.
- **API restrictions:** Google Picker API (Docs и Drive в списке не мешают).
- В ai-tools: `REACT_APP_GOOGLE_API_KEY` (Actions variable + `.env`). Ключ виден в коде страницы — это нормально,
  его защищают ограничения.
- Изменения ограничений применяются до 5 минут. «קוד מפתח ה-API אינו חוקי» в окне выбора — проверить пункты 1 и 4,
  для проверки временно снять Application restrictions.

## 5. Подтверждение сайта и проверка бренда (только чтобы показывать своё название/логотип)

1. https://search.google.com/search-console → Add property → **URL prefix** `https://<ai-tools>/` →
   HTML file. Файл `googledec045341cd72249.html` уже лежит в `ai-tools/public/` и отдаётся сайтом — **не удалять**.
2. Verify в Search Console (тем же Google-аккаунтом, что владеет проектом).
3. Через 24 часа снова отправить на проверку: Branding / Verification Center.

Без этого всё работает; Google просто показывает «kab.info» вместо названия.

## 6. Права и аккаунт

- Нужна роль Owner или Editor: https://console.cloud.google.com/iam-admin/iam?project=ai-tools-512410
- Если консоль не грузит страницы — проверить Google-аккаунт в правом верхнем углу (`&authuser=1` в ссылке),
  окно инкогнито без расширений.

## 7. Запись с сервера (общий документ трансляции)

ai-trns создаёт документ при создании live-задачи и дописывает в него (`ai_trns/api/gdoc.py`).

1. Тот же проект, API из раздела 1 (Docs, Drive).
2. Аккаунт — один из двух:
   - **Service account:** https://console.cloud.google.com/iam-admin/serviceaccounts?project=ai-tools-512410 →
     Create → Keys → Add key → JSON. У сервисного аккаунта нет своего Drive: нужна папка на **общем диске**, где он
     участник (Content manager), её id — в `AI_TRNS_GOOGLE_FOLDER`.
   - **Обычный аккаунт (authorized_user):** JSON с `client_id`, `client_secret`, `refresh_token` и scope
     `documents` + `drive` — документы лягут в «Мой диск» этого аккаунта.
3. JSON — на `ai-tools` (10.66.24.55) в `/home/ai-trns/google-credentials.json`, владелец `ai-trns`, права 600
   (копировать `scp`, не через чат).
4. `ai-trns-api.env`:
   ```
   AI_TRNS_GOOGLE_CREDENTIALS=/home/ai-trns/google-credentials.json
   AI_TRNS_GOOGLE_FOLDER=<id папки>            # обязательно для service account
   AI_TRNS_GOOGLE_DOC_ROLE=writer              # кто по ссылке может править; пусто — по доступу к папке
   ```
5. `systemctl restart ai-trns-api`.

## Где это в коде

- ai-tools: `src/api/gdoc.ts` (Google Identity Services, Docs/Drive API, Picker), `src/pages/Live.tsx`,
  `public/privacy.html`, `public/googledec045341cd72249.html`.
- ai-trns: `ai_trns/api/gdoc.py` (аккаунт сервера), `ai_trns/api/live.py` (`/live/{id}/doc`, события `docs`).
