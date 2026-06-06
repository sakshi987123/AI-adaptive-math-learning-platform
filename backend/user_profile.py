"""
User Profile & Learning History System
=======================================
Technologies used:
- Rule-based system: topic weakness detection from wrong answers
- Explainable AI (XAI): SHAP-style impact scoring per topic
- Generative AI: Gemini for personalized recommendations
- ANN-style: weighted scoring across multiple quiz attempts
- LSTM-style: sequential behavior tracking (improving/struggling/stable)
- SQLite: persistent storage (zero install)
"""

import sqlite3
import json
import os
import math
from datetime import datetime
from lstm_behavior import LSTMBehaviorModel

DB_PATH = os.path.join(os.path.dirname(__file__), 'learning_history.db')
LSTM_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'behavior_lstm.keras')
lstm_behavior_model = LSTMBehaviorModel(model_path=LSTM_MODEL_PATH, window_size=3)


# ── DATABASE SETUP ──────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            photo_url TEXT,
            disability_type TEXT DEFAULT 'blind',
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    # Quiz sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid TEXT NOT NULL,
            quiz_type TEXT,          -- 'diagnostic' or 'evaluation'
            disability TEXT,         -- 'blind' or 'deaf'
            level TEXT,              -- 'slow' or 'advanced'
            pre_score REAL,          -- diagnostic score (0-100)
            post_score REAL,         -- evaluation score (0-100)
            accuracy REAL,           -- percentage correct
            total_questions INTEGER,
            correct_answers INTEGER,
            wrong_answers INTEGER,
            avg_response_time REAL,
            max_streak INTEGER,
            attempts INTEGER DEFAULT 1,
            weak_topics TEXT,        -- JSON array
            strong_topics TEXT,      -- JSON array
            topic_impacts TEXT,      -- JSON object
            improvement_text TEXT,
            answers_detail TEXT,     -- JSON array of answer objects
            behavior TEXT,           -- 'improving' / 'struggling' / 'stable'
            timestamp TEXT,
            FOREIGN KEY (uid) REFERENCES users(uid)
        )
    ''')

    # Learning progress table (aggregated per user)
    c.execute('''
        CREATE TABLE IF NOT EXISTS learning_progress (
            uid TEXT PRIMARY KEY,
            current_level TEXT,
            total_sessions INTEGER DEFAULT 0,
            total_correct INTEGER DEFAULT 0,
            total_questions INTEGER DEFAULT 0,
            best_score REAL DEFAULT 0,
            latest_score REAL DEFAULT 0,
            score_history TEXT DEFAULT '[]',   -- JSON array of scores
            behavior_history TEXT DEFAULT '[]', -- JSON array of behaviors
            overall_behavior TEXT DEFAULT 'stable',
            weak_topics_history TEXT DEFAULT '{}', -- JSON: topic -> count
            last_updated TEXT,
            FOREIGN KEY (uid) REFERENCES users(uid)
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


# ── USER MANAGEMENT ──────────────────────────────────────────────────────────

def upsert_user(uid, name, email, photo_url='', disability_type='blind'):
    conn = get_db()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute('''
        INSERT INTO users (uid, name, email, photo_url, disability_type, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(uid) DO UPDATE SET
            name=excluded.name,
            email=excluded.email,
            photo_url=excluded.photo_url,
            disability_type=excluded.disability_type,
            updated_at=excluded.updated_at
    ''', (uid, name, email, photo_url, disability_type, now, now))
    conn.commit()
    conn.close()


def get_user(uid):
    conn = get_db()
    c = conn.cursor()
    row = c.execute('SELECT * FROM users WHERE uid=?', (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── BEHAVIOR ANALYSIS (actual LSTM + fallback) ──────────────────────────────

def analyze_behavior(score_history):
    """
    Predict behavior trend from score sequence.
    Primary: actual LSTM model.
    Fallback: deterministic heuristic if LSTM model is unavailable.
    Returns: 'improving' | 'struggling' | 'stable'
    """
    # Try actual LSTM first.
    try:
        return lstm_behavior_model.predict(score_history)
    except Exception:
        pass

    # Fallback heuristic for cold-start and environments without TensorFlow.
    if len(score_history) < 2:
        return 'stable'
    recent = score_history[-3:]
    diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
    avg_diff = sum(diffs) / len(diffs)
    if avg_diff >= 10:
        return 'improving'
    if avg_diff <= -10:
        return 'struggling'
    return 'stable'


def _collect_all_score_histories(cursor):
    rows = cursor.execute('SELECT score_history FROM learning_progress').fetchall()
    histories = []
    for r in rows:
        try:
            seq = json.loads(r['score_history']) if r['score_history'] else []
            if isinstance(seq, list) and len(seq) >= 3:
                histories.append([float(x) for x in seq])
        except Exception:
            continue
    return histories


# ── ANN-inspired weighted scoring ────────────────────────────────────────────

def compute_weighted_score(score_history):
    """
    ANN-inspired exponentially weighted average with recency bias.
    """
    if not score_history:
        return 0
    n = len(score_history)
    weights = [math.exp(0.3 * i) for i in range(n)]
    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(score_history, weights))
    return round(weighted_sum / total_weight, 1)


# ── TOPIC WEAKNESS TRACKING (Rule-based + XAI) ──────────────────────────────

def update_weak_topics_history(existing_json, new_weak_topics):
    """
    Rule-based: accumulate weak topic counts across sessions.
    XAI: topics with highest count = most impactful weaknesses.
    """
    history = json.loads(existing_json) if existing_json else {}
    for topic in new_weak_topics:
        history[topic] = history.get(topic, 0) + 1
    return json.dumps(history)


def get_persistent_weak_topics(weak_topics_history_json):
    """
    XAI: return topics that appeared as weak in 2+ sessions (persistent weakness).
    """
    history = json.loads(weak_topics_history_json) if weak_topics_history_json else {}
    return [t for t, count in history.items() if count >= 2]


# ── SAVE QUIZ SESSION ─────────────────────────────────────────────────────────

def save_quiz_session(uid, session_data):
    """
    Save a quiz session and update the user's learning progress.
    session_data keys: quiz_type, disability, level, score, total_questions,
                       correct, wrong, accuracy, avg_response_time, max_streak,
                       weak_topics, strong_topics, topic_impacts, improvement_text,
                       answers_detail, pre_score (optional)
    """
    conn = get_db()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()

    score = session_data.get('accuracy', 0)  # use accuracy % as the score

    # ── Insert session ──
    c.execute('''
        INSERT INTO quiz_sessions
        (uid, quiz_type, disability, level, pre_score, post_score, accuracy,
         total_questions, correct_answers, wrong_answers, avg_response_time,
         max_streak, weak_topics, strong_topics, topic_impacts, improvement_text,
         answers_detail, behavior, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        uid,
        session_data.get('quiz_type', 'evaluation'),
        session_data.get('disability', 'blind'),
        session_data.get('level', 'slow'),
        session_data.get('pre_score', 0),
        score,
        score,
        session_data.get('total_questions', 5),
        session_data.get('correct', 0),
        session_data.get('wrong', 0),
        session_data.get('avg_response_time', 0),
        session_data.get('max_streak', 0),
        json.dumps(session_data.get('weak_topics', [])),
        json.dumps(session_data.get('strong_topics', [])),
        json.dumps(session_data.get('topic_impacts', {})),
        session_data.get('improvement_text', ''),
        json.dumps(session_data.get('answers_detail', [])),
        'stable',  # will be updated below
        now
    ))

    # ── Update learning progress ──
    prog = c.execute('SELECT * FROM learning_progress WHERE uid=?', (uid,)).fetchone()

    if prog:
        score_history = json.loads(prog['score_history'])
        behavior_history = json.loads(prog['behavior_history'])
        weak_topics_history = prog['weak_topics_history']
    else:
        score_history = []
        behavior_history = []
        weak_topics_history = '{}'

    score_history.append(score)

    # Attempt periodic (re)training of real LSTM using all available histories.
    # This keeps training cost low while improving as data grows.
    try:
        if len(score_history) % 5 == 0:
            all_histories = _collect_all_score_histories(c)
            if len(score_history) >= 3:
                all_histories.append(score_history)
            lstm_behavior_model.train_if_possible(all_histories)
    except Exception:
        # Do not block user flow if training fails.
        pass

    behavior = analyze_behavior(score_history)
    behavior_history.append(behavior)
    weak_topics_history = update_weak_topics_history(
        weak_topics_history, session_data.get('weak_topics', [])
    )

    # Update session behavior
    c.execute('UPDATE quiz_sessions SET behavior=? WHERE uid=? AND timestamp=?',
              (behavior, uid, now))

    total_sessions = (prog['total_sessions'] if prog else 0) + 1
    total_correct = (prog['total_correct'] if prog else 0) + session_data.get('correct', 0)
    total_questions = (prog['total_questions'] if prog else 0) + session_data.get('total_questions', 5)
    best_score = max(prog['best_score'] if prog else 0, score)
    overall_behavior = analyze_behavior(score_history)

    if prog:
        c.execute('''
            UPDATE learning_progress SET
                current_level=?, total_sessions=?, total_correct=?, total_questions=?,
                best_score=?, latest_score=?, score_history=?, behavior_history=?,
                overall_behavior=?, weak_topics_history=?, last_updated=?
            WHERE uid=?
        ''', (
            session_data.get('level', 'slow'),
            total_sessions, total_correct, total_questions,
            best_score, score,
            json.dumps(score_history), json.dumps(behavior_history),
            overall_behavior, weak_topics_history, now, uid
        ))
    else:
        c.execute('''
            INSERT INTO learning_progress
            (uid, current_level, total_sessions, total_correct, total_questions,
             best_score, latest_score, score_history, behavior_history,
             overall_behavior, weak_topics_history, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            uid,
            session_data.get('level', 'slow'),
            total_sessions, total_correct, total_questions,
            best_score, score,
            json.dumps(score_history), json.dumps(behavior_history),
            overall_behavior, weak_topics_history, now
        ))

    conn.commit()
    conn.close()
    return {'behavior': behavior, 'overall_behavior': overall_behavior}


# ── GET FULL PROFILE ──────────────────────────────────────────────────────────

def get_full_profile(uid):
    conn = get_db()
    c = conn.cursor()

    user = c.execute('SELECT * FROM users WHERE uid=?', (uid,)).fetchone()
    if not user:
        conn.close()
        return None

    prog = c.execute('SELECT * FROM learning_progress WHERE uid=?', (uid,)).fetchone()

    sessions = c.execute(
        'SELECT * FROM quiz_sessions WHERE uid=? ORDER BY timestamp DESC LIMIT 10',
        (uid,)
    ).fetchall()

    conn.close()

    # Parse JSON fields
    progress_data = {}
    if prog:
        progress_data = dict(prog)
        progress_data['score_history'] = json.loads(prog['score_history'])
        progress_data['behavior_history'] = json.loads(prog['behavior_history'])
        progress_data['weak_topics_history'] = json.loads(prog['weak_topics_history'])
        progress_data['persistent_weak_topics'] = get_persistent_weak_topics(prog['weak_topics_history'])
        progress_data['weighted_score'] = compute_weighted_score(progress_data['score_history'])
        progress_data['overall_accuracy'] = round(
            (prog['total_correct'] / prog['total_questions'] * 100)
            if prog['total_questions'] > 0 else 0, 1
        )

    sessions_data = []
    for s in sessions:
        sd = dict(s)
        sd['weak_topics'] = json.loads(s['weak_topics'])
        sd['strong_topics'] = json.loads(s['strong_topics'])
        sd['topic_impacts'] = json.loads(s['topic_impacts'])
        sd['answers_detail'] = json.loads(s['answers_detail'])
        sessions_data.append(sd)

    return {
        'user': dict(user),
        'progress': progress_data,
        'recent_sessions': sessions_data
    }


# Initialize DB on import
init_db()
