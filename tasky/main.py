from __future__ import annotations
from typing import Optional, Generator, List, Dict, Tuple
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

from sqlalchemy import create_engine, select, func, and_
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship
from sqlalchemy import String, Integer, Text, Date, DateTime, ForeignKey

app = FastAPI(title="Tasky — задачи, привычки и архив")

BASE_DIR = Path(__file__).parent
INDEX_FILE = BASE_DIR / "templates" / "index.html"

# -------------------- Pydantic --------------------
class TaskCreate(BaseModel):
    title: str = Field(min_length=1)
    description: Optional[str] = None
    status: str = Field(default="todo", pattern=r"^(todo|in_progress|done)$")
    priority: int = Field(default=3, ge=1, le=5)  # 1 = высокий
    due_date: Optional[date] = None

class TaskPatch(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = Field(default=None, pattern=r"^(todo|in_progress|done)$")
    priority: Optional[int] = Field(default=None, ge=1, le=5)
    due_date: Optional[date] = None

class TaskOut(TaskCreate):
    id: int
    completed_at: Optional[date] = None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class HabitCreate(BaseModel):
    title: str = Field(min_length=1)
    period: str = Field(pattern=r"^(daily|weekly)$")
    target: int = Field(ge=1, le=20)

class HabitPatch(BaseModel):
    title: Optional[str] = None
    period: Optional[str] = Field(default=None, pattern=r"^(daily|weekly)$")
    target: Optional[int] = Field(default=None, ge=1, le=20)

class HabitOut(BaseModel):
    id: int
    title: str
    period: str
    target: int
    done: int
    model_config = ConfigDict(from_attributes=True)

# -------------------- SQLAlchemy модели --------------------
class Base(DeclarativeBase):
    pass

class TaskORM(Base):
    __tablename__ = "task"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text(), default=None)
    status: Mapped[str] = mapped_column(String(20), default="todo")
    priority: Mapped[int] = mapped_column(Integer, default=3)
    due_date: Mapped[Optional[date]] = mapped_column(Date(), default=None)
    completed_at: Mapped[Optional[date]] = mapped_column(Date(), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class HabitORM(Base):
    __tablename__ = "habit"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    period: Mapped[str] = mapped_column(String(10))  # daily|weekly
    target: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    logs: Mapped[List["HabitLogORM"]] = relationship(back_populates="habit", cascade="all, delete-orphan")

class HabitLogORM(Base):
    __tablename__ = "habit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    habit_id: Mapped[int] = mapped_column(ForeignKey("habit.id"))
    done_at: Mapped[date] = mapped_column(Date(), default=date.today)
    habit: Mapped[HabitORM] = relationship(back_populates="logs")

class ArchiveTaskORM(Base):
    __tablename__ = "archive_task"
    id: Mapped[int] = mapped_column(primary_key=True)
    original_id: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text())
    status: Mapped[str] = mapped_column(String(20))
    priority: Mapped[int] = mapped_column(Integer)
    due_date: Mapped[Optional[date]] = mapped_column(Date())
    completed_at: Mapped[Optional[date]] = mapped_column(Date())
    deleted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ArchiveHabitORM(Base):
    __tablename__ = "archive_habit"
    id: Mapped[int] = mapped_column(primary_key=True)
    original_id: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200))
    period: Mapped[str] = mapped_column(String(10))
    target: Mapped[int] = mapped_column(Integer)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

# --- Engine & create tables
engine = create_engine(
    "sqlite:///tasky.db",
    echo=False, future=True,
    connect_args={"check_same_thread": False},
)
Base.metadata.create_all(engine)

# --- мягкая миграция, если БД старая
def _soft_migrate():
    with engine.connect() as conn:
        def cols(table: str) -> set[str]:
            return {c[1] for c in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}

        task_cols = cols("task")
        if "completed_at" not in task_cols:
            conn.exec_driver_sql("ALTER TABLE task ADD COLUMN completed_at DATE")
        if "created_at" not in task_cols:
            conn.exec_driver_sql("ALTER TABLE task ADD COLUMN created_at DATETIME")
            conn.exec_driver_sql("UPDATE task SET created_at = CURRENT_TIMESTAMP")
        if "updated_at" not in task_cols:
            conn.exec_driver_sql("ALTER TABLE task ADD COLUMN updated_at DATETIME")
            conn.exec_driver_sql("UPDATE task SET updated_at = CURRENT_TIMESTAMP")

        # Архивные таблицы на всякий случай
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS archive_task (
          id INTEGER PRIMARY KEY,
          original_id INTEGER,
          title TEXT,
          description TEXT,
          status TEXT,
          priority INTEGER,
          due_date DATE,
          completed_at DATE,
          deleted_at DATETIME
        )""")
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS archive_habit (
          id INTEGER PRIMARY KEY,
          original_id INTEGER,
          title TEXT,
          period TEXT,
          target INTEGER,
          deleted_at DATETIME
        )""")
        conn.commit()
_soft_migrate()

def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

# -------------------- helpers --------------------
def _period_bounds(period: str) -> Tuple[date, date]:
    today = date.today()
    if period == "daily":
        return today, today
    start = today - timedelta(days=today.weekday())  # Mon..Sun
    end = start + timedelta(days=6)
    return start, end

def _habit_done_count(db: Session, habit: HabitORM) -> int:
    start, end = _period_bounds(habit.period)
    return db.scalar(
        select(func.count(HabitLogORM.id)).where(
            and_(HabitLogORM.habit_id == habit.id,
                 HabitLogORM.done_at >= start,
                 HabitLogORM.done_at <= end)
        )
    ) or 0

# -------------------- SCORE (экспонента) --------------------
B_TASK = 10
PRIO_W: Dict[int, float] = {1: 3.0, 2: 2.2, 3: 1.6, 4: 1.2, 5: 1.0}
EARLY_W, ONTIME_W, LATE_W = 1.2, 1.0, 0.5
HABIT_POOL = {"daily": 15.0, "weekly": 30.0}
OVER_MULT = 0.25
K = 60.0

def _deadline_weight(due: Optional[date], done: date) -> float:
    if not due: return ONTIME_W
    if done < due: return EARLY_W
    if done == due: return ONTIME_W
    return LATE_W

def _tasks_points_today(db: Session) -> float:
    today = date.today()
    tasks_today = db.execute(select(TaskORM).where(TaskORM.completed_at == today)).scalars().all()
    total = 0.0
    for t in tasks_today:
        total += B_TASK * PRIO_W.get(t.priority, 1.0) * _deadline_weight(t.due_date, today)
    return total

def _habits_points_current_period(db: Session) -> float:
    habits = db.execute(select(HabitORM)).scalars().all()
    s = 0.0
    for h in habits:
        done = _habit_done_count(db, h)
        pool = HABIT_POOL[h.period]
        per = pool / max(h.target, 1)
        base = min(done, h.target) * per
        over = max(0, done - h.target) * (per * OVER_MULT)
        s += base + over
    return s

def _score_percent(db: Session) -> float:
    from math import exp
    S = _tasks_points_today(db) + _habits_points_current_period(db)
    return round(100.0 * (1.0 - exp(-S / K)), 1)

# -------------------- API --------------------
@app.get("/")
def index():
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/score")
def score(db: Session = Depends(get_db)):
    return {"score": _score_percent(db)}

# ---- Tasks
@app.post("/tasks", response_model=TaskOut, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    data = payload.model_dump()
    now = datetime.utcnow()
    data["created_at"] = now
    data["updated_at"] = now
    if data.get("status") == "done":
        data["completed_at"] = date.today()
    task = TaskORM(**data)
    db.add(task); db.commit(); db.refresh(task)
    return task

@app.get("/tasks", response_model=List[TaskOut])
def list_tasks(db: Session = Depends(get_db), status: Optional[str] = None, q: Optional[str] = None):
    stmt = select(TaskORM)
    if status:
        stmt = stmt.where(TaskORM.status == status)
    if q:
        stmt = stmt.where((TaskORM.title.like(f"%{q}%")) | (TaskORM.description.like(f"%{q}%")))
    tasks = db.execute(stmt).scalars().all()
    # 1 — сверху
    tasks = sorted(tasks, key=lambda t: (t.priority, -t.id))
    return tasks

@app.patch("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, payload: TaskPatch, db: Session = Depends(get_db)):
    t = db.get(TaskORM, task_id)
    if not t: raise HTTPException(404, "Task not found")
    data = payload.model_dump(exclude_unset=True)
    if "status" in data:
        new_st = data["status"]
        if new_st == "done" and t.status != "done":
            data.setdefault("completed_at", date.today())
        if new_st != "done" and t.status == "done":
            data.setdefault("completed_at", None)
    for k, v in data.items():
        setattr(t, k, v)
    t.updated_at = datetime.utcnow()
    db.add(t); db.commit(); db.refresh(t)
    return t

@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(TaskORM, task_id)
    if not t: raise HTTPException(404, "Task not found")
    archived = ArchiveTaskORM(
        original_id=t.id,
        title=t.title,
        description=t.description,
        status=t.status,
        priority=t.priority,
        due_date=t.due_date,
        completed_at=t.completed_at,
        deleted_at=datetime.utcnow(),
    )
    db.add(archived)
    db.delete(t)
    db.commit()

@app.get("/archive/tasks", response_model=List[TaskOut])
def get_archive_tasks(db: Session = Depends(get_db)):
    records = db.execute(select(ArchiveTaskORM)).scalars().all()
    out: List[TaskOut] = []
    for r in records:
        out.append(TaskOut(
            id=r.original_id,
            title=r.title,
            description=r.description,
            status=r.status,
            priority=r.priority,
            due_date=r.due_date,
            completed_at=r.completed_at,
            created_at=r.deleted_at,
            updated_at=r.deleted_at
        ))
    return out

# ---- Habits
@app.post("/habits", response_model=HabitOut, status_code=201)
def create_habit(payload: HabitCreate, db: Session = Depends(get_db)):
    h = HabitORM(**payload.model_dump())
    db.add(h); db.commit(); db.refresh(h)
    return HabitOut(id=h.id, title=h.title, period=h.period, target=h.target, done=_habit_done_count(db, h))

@app.get("/habits", response_model=List[HabitOut])
def list_habits(db: Session = Depends(get_db)):
    habits = db.execute(select(HabitORM)).scalars().all()
    out: List[HabitOut] = []
    for h in habits:
        out.append(HabitOut(id=h.id, title=h.title, period=h.period, target=h.target, done=_habit_done_count(db, h)))
    out.sort(key=lambda x: (0 if x.period == "daily" else 1, -x.id))
    return out

@app.patch("/habits/{habit_id}", response_model=HabitOut)
def update_habit(habit_id: int, payload: HabitPatch, db: Session = Depends(get_db)):
    h = db.get(HabitORM, habit_id)
    if not h: raise HTTPException(404, "Habit not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(h, k, v)
    db.add(h); db.commit(); db.refresh(h)
    return HabitOut(id=h.id, title=h.title, period=h.period, target=h.target, done=_habit_done_count(db, h))

@app.delete("/habits/{habit_id}", status_code=204)
def delete_habit(habit_id: int, db: Session = Depends(get_db)):
    h = db.get(HabitORM, habit_id)
    if not h: raise HTTPException(404, "Habit not found")
    # архивируем привычку
    archived = ArchiveHabitORM(
        original_id=h.id,
        title=h.title,
        period=h.period,
        target=h.target,
        deleted_at=datetime.utcnow()
    )
    db.add(archived)
    # удаляем логи и саму привычку
    logs = db.execute(select(HabitLogORM).where(HabitLogORM.habit_id == h.id)).scalars().all()
    for lg in logs:
        db.delete(lg)
    db.delete(h)
    db.commit()

@app.post("/habits/{habit_id}/check", response_model=HabitOut)
def check_habit(habit_id: int, db: Session = Depends(get_db)):
    h = db.get(HabitORM, habit_id)
    if not h: raise HTTPException(404, "Habit not found")
    db.add(HabitLogORM(habit_id=h.id, done_at=date.today()))
    db.commit()
    return HabitOut(id=h.id, title=h.title, period=h.period, target=h.target, done=_habit_done_count(db, h))

@app.post("/habits/{habit_id}/uncheck", response_model=HabitOut)
def uncheck_habit(habit_id: int, db: Session = Depends(get_db)):
    h = db.get(HabitORM, habit_id)
    if not h: raise HTTPException(404, "Habit not found")
    start, end = _period_bounds(h.period)
    last = db.execute(
        select(HabitLogORM)
        .where(and_(HabitLogORM.habit_id == h.id,
                    HabitLogORM.done_at >= start,
                    HabitLogORM.done_at <= end))
        .order_by(HabitLogORM.id.desc())
    ).scalars().first()
    if last:
        db.delete(last); db.commit()
    return HabitOut(id=h.id, title=h.title, period=h.period, target=h.target, done=_habit_done_count(db, h))

@app.get("/archive/habits")
def get_archive_habits(db: Session = Depends(get_db)):
    records = db.execute(select(ArchiveHabitORM)).scalars().all()
    return [
        {
            "id": r.original_id,
            "title": r.title,
            "period": r.period,
            "target": r.target,
            "deleted_at": r.deleted_at.isoformat()
        } for r in records
    ]
