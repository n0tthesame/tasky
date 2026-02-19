Tasky — трекер задач и привычек
Клиент-серверное веб-приложение для управления задачами и привычками с расчётом индекса продуктивности (Score).

Стек:
- Python 3.13
- FastAPI, Uvicorn
- SQLAlchemy (ORM)
- Pydantic (валидация)
- SQLite
- Frontend: HTML + CSS (тёмная тема) + JavaScript (fetch API)

Возможности:
- CRUD для задач и привычек
- Архивирование удалённых задач/привычек (перенос в таблицы archive_*)
- Расчёт показателя продуктивности (Score)

Архитектура:
- UI: templates/index.html + JS (fetch) → запросы к REST API
- API: FastAPI
- БД: SQLite через SQLAlchemy
- Документация API: Swagger/OpenAPI

Формула Score: score = 100 * (1 - exp(-S / 60))
