from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Корень проекта
ROOT = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

# Доступ к событиям Google Calendar
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events"
]


def get_calendar_service():
    creds = None

    # Если token.json уже существует
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    # Если авторизация отсутствует или устарела
    if not creds or not creds.valid:

        # Обновляем токен
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        # Первая авторизация
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    "Не найден credentials.json в корне проекта"
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Сохраняем токен локально
        TOKEN_FILE.write_text(
            creds.to_json(),
            encoding="utf-8"
        )

    return build(
        "calendar",
        "v3",
        credentials=creds
    )


def test_connection():
    """
    Проверяет подключение к Google Calendar.
    """

    service = get_calendar_service()

    service.events().list(
        calendarId="primary",
        maxResults=1,
        singleEvents=True
    ).execute()

    print("Google Calendar подключён успешно!")
    print("Доступ к событиям получен.")


def create_event(
    title,
    start_time,
    end_time,
    description="",
    location=""
):
    """
    Создаёт событие в основном Google Calendar.
    """

    service = get_calendar_service()

    event = {
        "summary": title,
        "description": description,
        "location": location,
        "start": {
            "dateTime": start_time
        },
        "end": {
            "dateTime": end_time
        }
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event
    ).execute()

    return {
        "id": created_event.get("id"),
        "title": created_event.get("summary"),
        "link": created_event.get("htmlLink")
    }


def delete_event(event_id):
    """
    Удаляет событие из Google Calendar.
    """

    if not event_id:
        raise ValueError("Не указан ID события")

    service = get_calendar_service()

    service.events().delete(
        calendarId="primary",
        eventId=event_id
    ).execute()

    return {
        "id": event_id,
        "status": "deleted"
    }


if __name__ == "__main__":
    test_connection()