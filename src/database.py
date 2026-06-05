import sqlite3
from pathlib import Path

DB_PATH = Path("data/safedrive.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drowsiness_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT,
            operator_id TEXT,
            prediction TEXT,
            risk_level TEXT,
            confidence REAL,
            ear_mean REAL,
            ear_min REAL,
            mar_mean REAL,
            mar_max REAL,
            perclos REAL,
            blink_count INTEGER,
            yawn_count INTEGER,
            head_pitch REAL,
            face_missing_ratio REAL,
            event_type TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_event(event):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO drowsiness_events (
            timestamp,
            source,
            operator_id,
            prediction,
            risk_level,
            confidence,
            ear_mean,
            ear_min,
            mar_mean,
            mar_max,
            perclos,
            blink_count,
            yawn_count,
            head_pitch,
            face_missing_ratio,
            event_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event["timestamp"],
        event["source"],
        event["operator_id"],
        event["prediction"],
        event["risk_level"],
        event["confidence"],
        event["ear_mean"],
        event["ear_min"],
        event["mar_mean"],
        event["mar_max"],
        event["perclos"],
        event["blink_count"],
        event["yawn_count"],
        event["head_pitch"],
        event["face_missing_ratio"],
        event["event_type"]
    ))

    conn.commit()
    conn.close()