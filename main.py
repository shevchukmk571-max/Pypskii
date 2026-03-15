#!/usr/bin/env python3
"""
🚀 Pypsiki in Space — Главный модуль (бэкенд + точка входа)
"""
import sys, os, pickle, traceback, sqlite3, requests, io, numpy as np, pandas as pd, re
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import bcrypt, secrets
import config

# ===== БАЗА ДАННЫХ =====
@contextmanager
def get_db(db_path=None):
    db_path = db_path or config.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(db_path=None):
    db_path = db_path or config.DB_PATH
    with get_db(db_path) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL, first_name TEXT NOT NULL, last_name TEXT NOT NULL,
            role TEXT CHECK(role IN ('admin', 'user')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login TIMESTAMP)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_username ON users(username)")
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL, FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE)""")
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, input_features TEXT,
            predicted_class INTEGER, confidence REAL, actual_class INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id))""")
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO users (username, password_hash, first_name, last_name, role) VALUES (?, ?, ?, ?, ?)",
                     ("admin", hash_password("admin123"), "System", "Administrator", "admin"))
            print("✅ Создан админ: admin / admin123")

def get_user(username, db_path=None):
    with get_db(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        return c.fetchone()

def create_user(username, pwd_hash, first, last, role="user", db_path=None):
    with get_db(db_path) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash, first_name, last_name, role) VALUES (?, ?, ?, ?, ?)",
                 (username, pwd_hash, first, last, role))
        return c.lastrowid

def update_login(uid, db_path=None):
    with get_db(db_path) as conn:
        conn.cursor().execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (uid,))

def log_pred(uid, features, pred_class, conf, actual=None, db_path=None):
    with get_db(db_path) as conn:
        conn.cursor().execute("INSERT INTO predictions (user_id, input_features, predicted_class, confidence, actual_class) VALUES (?, ?, ?, ?, ?)",
                             (uid, features, pred_class, conf, actual))

# ===== АВТОРИЗАЦИЯ =====
def hash_password(pwd, rounds=12):
    return bcrypt.hashpw(pwd.encode(), bcrypt.gensalt(rounds)).decode()

def verify_password(pwd, pwd_hash):
    return bcrypt.checkpw(pwd.encode(), pwd_hash.encode())

def gen_token(uid, hours=24):
    token = secrets.token_urlsafe(32)
    expires = datetime.now() + timedelta(hours=hours)
    with get_db() as conn:
        conn.cursor().execute("INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)", (uid, token, expires))
    return token

def validate_token(token):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT u.*, s.expires_at FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > CURRENT_TIMESTAMP", (token,))
        r = c.fetchone()
        return dict(r) if r else None

def authenticate(username, password):
    user = get_user(username)
    if user and verify_password(password, user["password_hash"]):
        update_login(user["id"])
        return {"id": user["id"], "username": user["username"], "first_name": user["first_name"], 
                "last_name": user["last_name"], "role": user["role"], "token": gen_token(user["id"])}
    return None

# ===== ЗАГРУЗКА ДАННЫХ .NPZ =====
class DataNPZLoader:
    def __init__(self, url=None, local_path=None):
        self.url = url or config.DATA_URL
        self.local_path = Path(local_path) if local_path else config.DATA_FILE
    
    def download(self):
        if self.local_path.exists():
            return True
        for url in [self.url.replace('/d/','/download/'), self.url]:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                self.local_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.local_path, 'wb') as f:
                    f.write(r.content)
                return True
            except:
                continue
        return False
    
    def load(self):
        if not self.local_path.exists() and not self.download():
            raise FileNotFoundError(f"Не найден: {self.local_path}")
        data = np.load(self.local_path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    
    @staticmethod
    def normalize_signal_length(signals, target_length=None):
        """Надёжная нормализация длины сигналов"""
        print(f"  📊 Нормализация {len(signals)} сигналов...")
        
        if len(signals) == 0:
            return np.zeros((0, 100), dtype=np.float32), 100
        
        signal_list = []
        lengths = []
        for i, sig in enumerate(signals):
            try:
                arr = np.array(sig).flatten().astype(np.float32)
                if len(arr) == 0:
                    arr = np.zeros(100, dtype=np.float32)
                signal_list.append(arr)
                lengths.append(len(arr))
            except:
                signal_list.append(np.zeros(100, dtype=np.float32))
                lengths.append(100)
        
        if target_length is None:
            target_length = int(np.median(lengths))
            if target_length < 50:
                target_length = 100
        
        print(f"  📏 Целевая длина: {target_length} (мин: {min(lengths)}, макс: {max(lengths)}, медиана: {int(np.median(lengths))})")
        
        normalized = np.zeros((len(signal_list), target_length), dtype=np.float32)
        for i, sig in enumerate(signal_list):
            if len(sig) < target_length:
                normalized[i, :len(sig)] = sig
            elif len(sig) > target_length:
                x_old = np.linspace(0, 1, len(sig))
                x_new = np.linspace(0, 1, target_length)
                normalized[i] = np.interp(x_new, x_old, sig)
            else:
                normalized[i] = sig
        
        print(f"  ✅ Форма после нормализации: {normalized.shape}")
        return normalized, target_length
    
    def prepare(self, data):
        """Подготовка данных с оптимизацией"""
        print("📂 Определение формата данных...")
        
        if 'train_x' in data:
            X_train, y_train = data['train_x'], data['train_y']
            X_valid = data.get('valid_x', np.array([]))
            y_valid = data.get('valid_y', np.array([]))
        elif 'train' in data and isinstance(data['train'], dict):
            X_train = data['train'].get('x') or data['train'].get('X')
            y_train = data['train'].get('y') or data['train'].get('Y')
            X_valid = data.get('valid', {}).get('x') or data.get('valid', {}).get('X', np.array([]))
            y_valid = data.get('valid', {}).get('y') or data.get('valid', {}).get('Y', np.array([]))
        else:
            keys = list(data.keys())
            X_train = data.get(keys[0])
            y_train = data.get(keys[1]) if len(keys) > 1 else None
            X_valid = data.get(keys[2]) if len(keys) > 2 else np.array([])
            y_valid = data.get(keys[3]) if len(keys) > 3 else np.array([])
        
        print(f"📊 Размер train_x: {len(X_train) if X_train is not None else 0}")
        print(f"📊 Размер valid_x: {len(X_valid) if len(X_valid) > 0 else 0}")
        
        print("🔧 Нормализация длины сигналов...")
        X_train, target_len = self.normalize_signal_length(X_train)
        
        if len(X_valid) > 0:
            X_valid, _ = self.normalize_signal_length(X_valid, target_length=target_len)
        else:
            X_valid = np.zeros((0, target_len), dtype=np.float32)
        
        print(f"✅ Форма X_train: {X_train.shape}")
        print(f"✅ Форма X_valid: {X_valid.shape}")
        
        if y_train is not None:
            y_train = np.array(y_train).flatten()
        else:
            y_train = np.array([])
        
        if len(y_valid) > 0:
            y_valid = np.array(y_valid).flatten()
        else:
            y_valid = np.array([])
        
        def extract_class(label):
            label = str(label)
            match = re.search(r'[0-9a-f]{32}(.+)', label, re.IGNORECASE)
            return match.group(1) if match else label
        
        if len(y_train) > 0 and isinstance(y_train[0], (str, np.str_)):
            sample = str(y_train[0])
            if re.match(r'^[0-9a-f]{32}', sample, re.IGNORECASE):
                print("🔧 Извлечение классов из хешей...")
                y_train = np.array([extract_class(l) for l in y_train])
        
        if len(y_valid) > 0 and isinstance(y_valid[0], (str, np.str_)):
            sample = str(y_valid[0])
            if re.match(r'^[0-9a-f]{32}', sample, re.IGNORECASE):
                y_valid = np.array([extract_class(l) for l in y_valid])
        
        encoder = LabelEncoder()
        all_labels = []
        if len(y_train) > 0:
            all_labels.extend(y_train)
        if len(y_valid) > 0:
            all_labels.extend(y_valid)
        
        if len(all_labels) > 0:
            print(f"🔧 Кодирование {len(all_labels)} меток...")
            encoder.fit(all_labels)
            if len(y_train) > 0:
                y_train = encoder.transform(y_train)
            if len(y_valid) > 0:
                y_valid = encoder.transform(y_valid)
        
        print(f"✅ Классы: {encoder.classes_}")
        print(f"✅ Train: {len(y_train)} примеров, Valid: {len(y_valid)} примеров")
        
        return X_train, y_train, X_valid, y_valid, encoder

# ===== ПРЕДОБРАБОТКА СИГНАЛОВ =====
class SignalPreprocessor:
    def __init__(self):
        self.mean = self.std = None
        self.fitted = False
        self.target_len = None
    
    def fit(self, X):
        X = np.atleast_2d(X)
        self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0) + 1e-8
        self.target_len = X.shape[1] if X.ndim == 2 else None
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Сначала fit()!")
        X = np.atleast_2d(X)
        if self.target_len and X.shape[1] != self.target_len:
            X = np.array([np.interp(np.linspace(0,1,self.target_len), np.linspace(0,1,len(s)), s) for s in X])
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    @staticmethod
    def remove_noise(X, thr=0.01):
        Xc = X.copy()
        Xc[np.abs(Xc) < thr] = 0
        return Xc
    
    @staticmethod
    def extract_fft(X, n=20):
        return np.array([np.abs(np.fft.fft(s))[:n] for s in X])
    
    @staticmethod
    def extract_stats(X):
        feats = []
        for s in X:
            n = len(s)
            if n == 0:
                feats.append(np.zeros(11))
                continue
            try:
                sk, kt = (stats.skew(s) if n>2 else 0), (stats.kurtosis(s) if n>3 else 0)
            except:
                sk = kt = 0
            feats.append(np.array([np.mean(s), np.std(s), np.min(s), np.max(s), np.median(s), 
                                  np.ptp(s), np.sqrt(np.mean(s**2)), sk, kt, 
                                  np.percentile(s,25), np.percentile(s,75)]))
        return np.array(feats)
    
    @staticmethod
    def smooth(X, ws=5):
        if ws < 2:
            return X
        k = np.ones(ws)/ws
        return np.array([np.convolve(s, k, mode='same') for s in X])

# ===== НЕЙРОСЕТЬ =====
class AlienClassifier:
    def __init__(self):
        self.model = None
        self.prep = SignalPreprocessor()
        self.encoder = None
        self.trained = False
        self.history = []
        self.cfg = {'use_fft':True, 'use_stats':True, 'remove_noise':True, 'smooth':False, 
                   'noise_thr':0.01, 'fft_n':30, 'smooth_ws':5}
    
    def preprocess(self, X, train=True):
        Xp = X.astype(float).copy()
        if self.cfg['remove_noise']:
            Xp = self.prep.remove_noise(Xp, self.cfg['noise_thr'])
        if self.cfg['smooth']:
            Xp = self.prep.smooth(Xp, self.cfg['smooth_ws'])
        feats = [Xp]
        if self.cfg['use_fft']:
            feats.append(self.prep.extract_fft(Xp, self.cfg['fft_n']))
        if self.cfg['use_stats']:
            feats.append(self.prep.extract_stats(Xp))
        Xf = np.hstack(feats)
        return self.prep.fit_transform(Xf) if train else self.prep.transform(Xf)
    
    def build(self, inp, classes):
        m = keras.Sequential([
            layers.Input(shape=(inp,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(classes, activation='softmax')
        ])
        m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return m
    
    def train(self, Xtr, ytr, Xv=None, yv=None, on_epoch=None):
        """Обучение модели — все 20 эпох пройдут"""
        Xtr_p = self.preprocess(Xtr, True)
        Xv_p = self.preprocess(Xv, False) if Xv is not None and len(Xv)>0 else None
        val = (Xv_p, yv) if Xv_p is not None else None
        self.model = self.build(Xtr_p.shape[1], len(np.unique(ytr)))
        
        class ProgCb(callbacks.Callback):
            def __init__(self, cb):
                self.cb = cb
            def on_epoch_end(self, ep, logs=None):
                if self.cb:
                    self.cb({'epoch':ep+1, 'acc':logs.get('accuracy'), 'val_acc':logs.get('val_accuracy')})
        
        hist = self.model.fit(Xtr_p, ytr, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                             validation_split=config.VALIDATION_SPLIT if not val else 0,
                             validation_data=val, callbacks=[ProgCb(on_epoch)], verbose=0)
        self.history = hist.history
        self.trained = True
        return hist.history
    
    def predict(self, X):
        if not self.trained:
            raise RuntimeError("Модель не обучена!")
        Xp = self.preprocess(X, False)
        preds = self.model.predict(Xp, verbose=0)
        return np.argmax(preds, axis=1), np.max(preds, axis=1), preds
    
    def evaluate(self, X, y):
        return self.model.evaluate(self.preprocess(X, False), y, verbose=0, return_dict=True)
    
    def save(self, path=None):
        path = Path(path) if path else config.MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.h5")
        with open(path / "prep.pkl", 'wb') as f:
            pickle.dump({'mean':self.prep.mean, 'std':self.prep.std, 'fitted':self.prep.fitted, 'tlen':self.prep.target_len}, f)
        with open(path / "cfg.pkl", 'wb') as f:
            pickle.dump(self.cfg, f)
        with open(path / "hist.pkl", 'wb') as f:
            pickle.dump(self.history, f)
    
    def load(self, path=None):
        path = Path(path) if path else config.MODELS_DIR
        if not all((path/f).exists() for f in ["model.h5","prep.pkl","cfg.pkl"]):
            return False
        try:
            self.model = keras.models.load_model(path / "model.h5")
            with open(path / "prep.pkl", 'rb') as f:
                d = pickle.load(f)
                self.prep.mean, self.prep.std, self.prep.fitted, self.prep.target_len = d['mean'], d['std'], d['fitted'], d.get('tlen')
            with open(path / "cfg.pkl", 'rb') as f:
                self.cfg = pickle.load(f)
            if (path / "hist.pkl").exists():
                with open(path / "hist.pkl", 'rb') as f:
                    self.history = pickle.load(f)
            self.trained = True
            return True
        except:
            return False

# ===== АНАЛИТИКА (matplotlib) =====
def plot_accuracy(hist, title="Точность обучения"):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5), facecolor='#16213e')
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
    epochs = range(1, len(hist['accuracy'])+1)
    for ax, key, label in [(ax1,'accuracy','Точность'), (ax2,'loss','Потери')]:
        ax.plot(epochs, hist[key], 'b-', label='Train', linewidth=2)
        if f'val_{key}' in hist:
            ax.plot(epochs, hist[f'val_{key}'], 'r--', label='Valid', linewidth=2)
        ax.set_xlabel('Эпоха', color='white')
        ax.set_ylabel(label, color='white')
        ax.set_title(label, color='#00d9ff')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.tick_params(colors='white')
        for s in ax.spines.values():
            s.set_color('white')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')
    plt.tight_layout()
    return fig

def plot_distribution(y, names=None, title="Распределение классов"):
    import matplotlib.pyplot as plt
    u, c = np.unique(y, return_counts=True)
    labels = [names[i] if names and i<len(names) else f"Class {i}" for i in u] if names else [f"Класс {i}" for i in u]
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#16213e')
    bars = ax.bar(labels, c, color='#00d9ff', edgecolor='white')
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{int(b.get_height())}', ha='center', va='bottom', color='white', fontsize=9)
    ax.set_xlabel('Класс', color='white')
    ax.set_ylabel('Количество', color='white')
    ax.set_title(title, color='#00d9ff')
    ax.tick_params(axis='x', rotation=45, colors='white')
    ax.grid(axis='y', alpha=0.3)
    for s in ax.spines.values():
        s.set_color('white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    plt.tight_layout()
    return fig

def plot_predictions(y_true, y_pred, conf=None, title="Точность предсказаний"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,5), facecolor='#16213e')
    if conf is not None:
        sc = ax.scatter(range(len(y_true)), y_pred, c=conf, cmap='viridis', s=30, alpha=0.7, edgecolors='white')
        plt.colorbar(sc, ax=ax, label='Уверенность')
    else:
        colors = ['#00c853' if t==p else '#f44336' for t,p in zip(y_true, y_pred)]
        ax.scatter(range(len(y_true)), y_pred, c=colors, s=30, alpha=0.7)
    acc = np.mean(y_true==y_pred)*100
    ax.set_xlabel('Пример', color='white')
    ax.set_ylabel('Класс', color='white')
    ax.set_title(f'{title}\nТочность: {acc:.1f}%', color='#00d9ff')
    ax.grid(axis='y', alpha=0.3)
    for s in ax.spines.values():
        s.set_color('white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    plt.tight_layout()
    return fig

def plot_top(y, n=5, names=None, title="Топ-5 классов"):
    import matplotlib.pyplot as plt
    u, c = np.unique(y, return_counts=True)
    idx = np.argsort(c)[::-1][:n]
    labels = [names[i] if names and i<len(names) else f"Class {i}" for i in u[idx]][::-1]
    fig, ax = plt.subplots(figsize=(10,6), facecolor='#16213e')
    ax.barh(labels, c[idx][::-1], color=['#00d9ff','#00b8d9','#0097a7','#00838f','#006064'], edgecolor='white')
    ax.set_xlabel('Количество', color='white')
    ax.set_title(title, color='#00d9ff')
    ax.grid(axis='x', alpha=0.3)
    for s in ax.spines.values():
        s.set_color('white')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('white')
    plt.tight_layout()
    return fig

# ===== ТОЧКА ВХОДА =====
def main():
    init_db()
    from frontend import run_app
    run_app()

if __name__ == "__main__":
    main()
