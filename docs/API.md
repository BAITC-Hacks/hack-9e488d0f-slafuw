# Контракт локального API

Запуск: `python -m eventmatch serve`, base URL `http://127.0.0.1:8000`.

## `POST /recommend`

### Вход

```json
{
  "city": "Алматы",
  "event_date": "2026-10-12",
  "event_format": "корпоратив",
  "category": "Ведущий",
  "budget_kzt": 1300000,
  "duration_hours": 6,
  "language": "русский"
}
```

| Поле | Тип | Правила |
|---|---|---|
| `city` | string | Обязательно, непустое; выбирает город каталога |
| `event_date` | string | Обязательно, точное `YYYY-MM-DD`, 23.09–31.12.2026 включительно |
| `event_format` | string | свадьба / той / корпоратив / конференция / юбилей / день рождения |
| `category` | string | Обязательно, непустое; точная категория, не поиск по подстроке |
| `budget_kzt` | integer | Обязательно, положительное; бюджет на этого подрядчика, ₸ за мероприятие |
| `duration_hours` | number или null | Необязательно, положительное конечное число, допустимы дробные часы |
| `language` | string или null | Необязательно: русский / казахский / английский |

Неуказанные опциональные поля эквивалентны `null`. Неизвестные поля, boolean вместо числа, строка вместо бюджета и даты вне календаря отклоняются. Неизвестные город или категория дают HTTP 422; `no_category_in_city` означает отсутствие сочетания двух известных значений. Значения предлагает `/metadata`.

Поддерживается только один язык. Обязательная двуязычность — отдельное расширение контракта на массив с AND-семантикой, а не запись `русский/казахский` в текущее поле.

### Успешный HTTP-ответ

Все три бизнес-исхода возвращают **HTTP 200**. Пустая выдача — валидный результат, не серверная ошибка.

| Поле | Назначение |
|---|---|
| `status` | `matched`, `no_category_in_city`, `no_match` |
| `summary` | Понятная сводка количества и причин |
| `request` | Нормализованный запрос |
| `cards` | От 0 до 3 карточек, уже в стабильном порядке |
| `diagnostics` | Круг кандидатов, исключения, причинная роль даты |
| `suggestions` | Проверенные изменения одного поля для `no_match` |
| `price_note` | Значение цены «от» |
| `meta` | Hash данных, версия политики, fingerprint, режим объяснений |

Активный API также возвращает `result_id`, `normalized_request` (алиас `request`),
`metadata` (алиас `meta`), `session_id`, `trace_id`, `actions`, `duration_ms`.
Карточка дополнена `facts[]` с `fact_id`, `available_evidence[]` с `evidence_id` и
`profile_id`, `selected_sources` и тегами проверенных условий. `metadata.ranking`
фиксирует режим и сведения NVIDIA-артефакта; `metadata.openai` отражает фактические
попытки/успехи API. Число успешных обращений не равно оценке качества.

Карточка:

```text
id, name, category, categories[], city, price_from_kzt
explanation                     # 1–2 предложения
evidence[]                      # структурированные основания и точная цитата
ranking                         # сработавшие корни, price/id tie-break
provenance                      # synthetic, city_imputed, price_imputed, labels[]
```

Снимок legacy-ядра: [01-dense.json](../examples/responses/01-dense.json). Активный сервер добавляет поля агента и собирает объяснения по проверенному плану; его live-ответ можно получить командой `python -m eventmatch agent --request examples/01-dense.json`.

### Диагностика

```text
cohort_count                    # профилей в городе и категории ДО условий
eligible_count                  # проходят ВСЕ условия
shown_count                     # min(3, eligible_count)
primary_exclusions              # взаимоисключающий учёт по первой причине
all_exclusions                  # все нарушения; сумма может пересекаться
blocked_only_by_date            # нарушают ТОЛЬКО условие даты
eligible_ignoring_date           # прошли бы при отключении проверки даты
rejected[]: {id, reasons[]}      # полный список исключённых, упорядочен по id
```

Коды причин: `busy`, `budget`, `format`, `language`, `duration`. Последовательность определения первой причины зафиксирована именно в этом порядке.

### Пустая категория

Для `Астана + Декоратор`:

```json
{
  "status": "no_category_in_city",
  "summary": "В каталоге города «Астана» нет категории «Декоратор».",
  "cards": []
}
```

Здесь показаны только ключевые поля; полный ответ также содержит нормализованный запрос, диагностику и метаданные.

### Нет проходящих по условиям

Для `examples/04-busy.json`: `cohort_count=10`, `eligible_count=0`, `eligible_ignoring_date=3`, `blocked_only_by_date=3`. Это **не** означает «все десять заняты»: у остальных есть дополнительные нарушения, а один профиль свободен, но не берёт формат.

Подсказка:

```json
{
  "field": "event_date",
  "value": "2026-12-13",
  "eligible_count": 1,
  "message": "На 2026-12-13 пройдут 1; остальные условия те же"
}
```

## Ошибки

Неверный JSON, неверные типы, пропущенные обязательные поля, дата вне календаря:

```json
{
  "error": "invalid_request",
  "message": "event_date: календарь известен только с 2026-09-23 по 2026-12-31"
}
```

HTTP **422**. Локальный адаптер ограничивает тело запроса 32 768 байтами. Неизвестный маршрут возвращает **404** с текстовым объяснением в JSON. Техническая невозможность подбора или разбора текста — HTTP **503**, `status=technical_error`, не `no_match`.

CLI при ошибке возвращает JSON с `error=invalid_input` и ненулевой код завершения `2`.

## Служебные маршруты

- `GET /` — минимальная HTML-форма и демокнопки.
- `GET /health` — `status`, число профилей, SHA-256 датасета, версия политики.
- `GET /metadata` — города, категории, форматы, языки, границы календаря, версии каталога/алгоритма/артефакта и доступность настройки OpenAI.
- `GET /traces/{trace_id}` — сохранённый журнал фактических действий.

## Диалог и инструменты

### `POST /chat`

```json
{"message": "А если на 13 октября?", "session_id": "ID из предыдущего ответа"}
```

Для нового диалога `session_id` опускается или равен `null`. Обязательные пропуски
дают `status=clarification`, `missing_fields` и один вопрос. Остальные ответы:
три бизнес-исхода, `compared`, `explanation`, `technical_error`. `order` содержит
текущее структурированное состояние. `explanation.result` — предыдущий ответ
с его исходными объяснениями и журналом. Ключи API через этот endpoint не принимаются.

`POST /recommend` и `/compare-dates` возвращают `session_id`; для продолжения того же
диалога из формы можно передать его в заголовке `X-EventMatch-Session`.

### `POST /compare-dates`

```text
{request: полный заказ, date_a: "2026-10-12", date_b: "2026-10-13"}
```

Ответ: `status=compared`, `results[2]`, `appeared_ids`, `disappeared_ids`, `changes`.
В `changes`: `availability_changed`, `top3_displacement`, `other_constraints` или
`unchanged`, все нарушения и допустимость на каждой дате. Остальные поля заказа
между датами одинаковы. Бюджетные/календарные подсказки не применяются молча.

### `POST /finalize-result`

```text
{result_id: "сохранённый ID", explanation_plan: {cards: [
  {profile_id: "ID из результата", fact_ids: ["разрешённый fact_id"], evidence_id: "разрешённый evidence_id"}
]}}
```

План должен перечислять ровно исходные карточки. Для каждой — 1–2 факта и одна
цитата либо `null`. Backend проверяет принадлежность источников и допуск, возвращает
карточки в исходном порядке. Неизвестные источники/результаты — HTTP 422.

Function calling регистрирует те же операции под именами `get_catalog_metadata`,
`recommend`, `compare_dates`, `finalize_result`. Детали — [PROVIDERS.md](PROVIDERS.md).

## Примеры вызова

PowerShell 5.1:

```powershell
$body = [System.Text.Encoding]::UTF8.GetBytes('{"city":"Алматы","event_date":"2026-10-12","event_format":"корпоратив","category":"Ведущий","budget_kzt":1300000,"duration_hours":6,"language":"русский"}')
Invoke-RestMethod -Uri "http://127.0.0.1:8000/recommend" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
```

curl в Linux/macOS, из корня репозитория:

```sh
curl http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  --data-binary @examples/01-dense.json
```

В Windows для установленного curl используйте `curl.exe`, чтобы не попасть в PowerShell-alias. Для платформонезависимой демонстрации достаточно CLI.
