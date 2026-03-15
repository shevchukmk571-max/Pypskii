import sys
import os
import pickle
import traceback
import pandas as pd
import numpy as np
from scipy import stats  # ✅ Добавлено для skew и kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QFileDialog,
    QCheckBox, QGroupBox, QProgressBar, QScrollArea, QFrame,
    QSizePolicy, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# СТИЛИ
# ==========================================
STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
    color: #eee;
}

QWidget {
    background-color: #1a1a2e;
    color: #eee;
    font-family: 'Segoe UI', Arial, sans-serif;
}

QLabel#title {
    font-size: 24px;
    font-weight: bold;
    color: #00d9ff;
    padding: 10px;
}

QLabel#subtitle {
    font-size: 14px;
    color: #888;
    padding: 5px;
}

QGroupBox {
    font-weight: bold;
    color: #00d9ff;
    border: 2px solid #16213e;
    border-radius: 10px;
    margin-top: 15px;
    padding-top: 15px;
    background-color: #16213e;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 15px;
    padding: 0 10px;
    color: #00d9ff;
}

QPushButton {
    background-color: #0f3460;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 25px;
    font-size: 14px;
    font-weight: bold;
    min-width: 150px;
}

QPushButton:hover {
    background-color: #1a508b;
}

QPushButton:pressed {
    background-color: #00d9ff;
    color: #1a1a2e;
}

QPushButton:disabled {
    background-color: #333;
    color: #666;
}

QPushButton#primary {
    background-color: #00d9ff;
    color: #1a1a2e;
}

QPushButton#primary:hover {
    background-color: #00b8d9;
}

QPushButton#success {
    background-color: #00c853;
    color: white;
}

QPushButton#success:hover {
    background-color: #00a843;
}

QLineEdit {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 8px;
    padding: 10px;
    font-size: 13px;
    color: #eee;
}

QLineEdit:focus {
    border: 2px solid #00d9ff;
}

QTextEdit {
    background-color: #0f0f1a;
    border: 2px solid #16213e;
    border-radius: 8px;
    padding: 10px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    color: #00ff88;
}

QCheckBox {
    color: #eee;
    font-size: 13px;
    spacing: 8px;
    padding: 5px;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 5px;
    border: 2px solid #0f3460;
    background-color: #16213e;
}

QCheckBox::indicator:checked {
    background-color: #00d9ff;
    border: 2px solid #00d9ff;
}

QProgressBar {
    background-color: #16213e;
    border: none;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #00d9ff;
    border-radius: 8px;
}

QLabel#status {
    padding: 10px;
    border-radius: 8px;
    font-weight: bold;
}

QLabel#status_ready {
    background-color: #00c853;
    color: white;
}

QLabel#status_training {
    background-color: #ff9800;
    color: white;
}

QLabel#status_error {
    background-color: #f44336;
    color: white;
}

QFrame#line {
    background-color: #0f3460;
    max-height: 2px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #16213e;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #0f3460;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #00d9ff;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

# ==========================================
# ПОТОК ОБУЧЕНИЯ
# ==========================================
class TrainingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log = pyqtSignal(str)
    
    def __init__(self, classifier, filepath):
        super().__init__()
        self.classifier = classifier
        self.filepath = filepath
    
    def run(self):
        try:
            success, message = self.classifier.train(self.filepath, self)
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, traceback.format_exc())


# ==========================================
# ПРЕДОБРАБОТКА (NumPy + SciPy)
# ==========================================
class SignalPreprocessor:
    def __init__(self):
        self.mean_values = None
        self.std_values = None
        self.is_fitted = False
    
    def fit(self, X):
        self.mean_values = np.mean(X, axis=0)
        self.std_values = np.std(X, axis=0) + 1e-8
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()!")
        return (X - self.mean_values) / self.std_values
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @staticmethod
    def remove_noise(X, threshold=0.1):
        X_clean = X.copy()
        X_clean[np.abs(X_clean) < threshold] = 0
        return X_clean
    
    @staticmethod
    def extract_fft_features(X, n_components=10):
        fft_features = []
        for sample in X:
            fft_result = np.fft.fft(sample)
            amplitudes = np.abs(fft_result[:n_components])
            fft_features.append(amplitudes)
        return np.array(fft_features)
    
    @staticmethod
    def extract_statistical_features(X):
        """
        Вычисляет статистические признаки для каждого сигнала.
        ✅ ИСПРАВЛЕНО: используем scipy.stats для skew и kurtosis
        """
        features = []
        for sample in X:
            try:
                skew_val = stats.skew(sample) if len(sample) > 2 else 0
                kurt_val = stats.kurtosis(sample) if len(sample) > 3 else 0
            except:
                skew_val = 0
                kurt_val = 0
            
            stats_array = np.array([
                np.mean(sample),
                np.std(sample),
                np.min(sample),
                np.max(sample),
                np.median(sample),
                np.ptp(sample),  # Размах (max - min)
                np.sqrt(np.mean(sample**2)),  # Среднеквадратичное значение
                skew_val,      # ✅ Асимметрия (scipy.stats)
                kurt_val,      # ✅ Эксцесс (scipy.stats)
                np.percentile(sample, 25),
                np.percentile(sample, 75),
            ])
            features.append(stats_array)
        return np.array(features)
    
    @staticmethod
    def smooth_signal(X, window_size=3):
        if window_size < 2:
            return X
        kernel = np.ones(window_size) / window_size
        smoothed = []
        for sample in X:
            smoothed_sample = np.convolve(sample, kernel, mode='same')
            smoothed.append(smoothed_sample)
        return np.array(smoothed)
    
    def save(self, path='preprocessor.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'mean_values': self.mean_values,
                'std_values': self.std_values,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, path='preprocessor.pkl'):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean_values = data['mean_values']
            self.std_values = data['std_values']
            self.is_fitted = data['is_fitted']
        return True


# ==========================================
# КЛАССИФИКАТОР
# ==========================================
class AlienSignalClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = SignalPreprocessor()
        self.is_trained = False
        self.feature_count = 0
        self.config = {
            'use_fft': False,
            'use_stats': True,
            'remove_noise': True,
            'smooth_signal': False,
            'noise_threshold': 0.01,
            'fft_components': 10,
            'smooth_window': 3
        }

    def load_and_clean_data(self, filepath):
        df = pd.read_csv(filepath)
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
        self.feature_count = X.shape[1]
        y_encoded = self.label_encoder.fit_transform(y_raw)
        return X, y_encoded, len(self.label_encoder.classes_)

    def preprocess_data(self, X, is_training=True):
        X_processed = X.copy()
        original_features = X_processed.shape[1]
        
        if self.config['remove_noise']:
            X_processed = self.preprocessor.remove_noise(X_processed, self.config['noise_threshold'])
        
        if self.config['smooth_signal']:
            X_processed = self.preprocessor.smooth_signal(X_processed, self.config['smooth_window'])
        
        if self.config['use_fft']:
            fft_features = self.preprocessor.extract_fft_features(X_processed, self.config['fft_components'])
            X_processed = np.hstack([X_processed, fft_features])
        
        if self.config['use_stats']:
            # ✅ ИСПРАВЛЕНО: берём только оригинальные признаки для статистики
            stat_features = self.preprocessor.extract_statistical_features(X[:, :original_features])
            X_processed = np.hstack([X_processed, stat_features])
        
        if is_training:
            X_processed = self.preprocessor.fit_transform(X_processed)
        else:
            X_processed = self.preprocessor.transform(X_processed)
        
        return X_processed

    def build_model(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, filepath, thread=None):
        try:
            X, y, num_classes = self.load_and_clean_data(filepath)
            
            if thread:
                thread.log.emit("🔧 Предобработка данных через NumPy...")
            X_processed = self.preprocess_data(X, is_training=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )
            
            if thread:
                thread.log.emit("🧠 Построение нейронной сети...")
            self.model = self.build_model(X_train.shape[1], num_classes)
            
            if thread:
                thread.log.emit("🚀 Начало обучения...")
            
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, thread, epochs):
                    self.thread = thread
                    self.epochs = epochs
                    self.current_epoch = 0
                
                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch
                    if self.thread:
                        progress = int((epoch / self.epochs) * 100)
                        self.thread.progress.emit(progress, f"Эпоха {epoch+1}/{self.epochs}")
                
                def on_epoch_end(self, epoch, logs=None):
                    if self.thread:
                        acc = logs.get('accuracy', 0) * 100
                        val_acc = logs.get('val_accuracy', 0) * 100
                        self.thread.log.emit(f"✓ Эпоха {epoch+1}: точность={acc:.1f}%, валидация={val_acc:.1f}%")
            
            progress_callback = ProgressCallback(thread, 20)
            
            history = self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                callbacks=[progress_callback],
                verbose=0
            )
            
            loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
            
            self.is_trained = True
            
            self.save_model('alien_model.h5')
            self.save_encoder('encoder.pkl')
            self.preprocessor.save('preprocessor.pkl')
            self.save_config('config.pkl')
            
            return True, f"Обучение завершено! Точность: {acc * 100:.2f}%"
            
        except Exception as e:
            return False, traceback.format_exc()

    def save_model(self, path):
        if self.model:
            self.model.save(path)
    
    def save_encoder(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def save_config(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.config, f)

    def load_artifacts(self):
        required_files = ['alien_model.h5', 'encoder.pkl', 'preprocessor.pkl', 'config.pkl']
        if not all(os.path.exists(f) for f in required_files):
            return False
        try:
            self.model = keras.models.load_model('alien_model.h5')
            with open('encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.preprocessor.load('preprocessor.pkl')
            with open('config.pkl', 'rb') as f:
                self.config = pickle.load(f)
            self.is_trained = True
            return True
        except Exception:
            return False

    def predict_signal(self, signal_features):
        if not self.is_trained:
            return None, None, 0
        
        signal_array = np.array(signal_features).reshape(1, -1)
        signal_processed = self.preprocess_data(signal_array, is_training=False)
        
        prediction_prob = self.model.predict(signal_processed, verbose=0)[0]
        predicted_idx = np.argmax(prediction_prob)
        
        class_name = self.label_encoder.inverse_transform([predicted_idx])[0]
        class_number_for_user = predicted_idx + 1
        confidence = prediction_prob[predicted_idx] * 100
        
        return class_name, class_number_for_user, confidence


# ==========================================
# ГЛАВНОЕ ОКНО
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.classifier = AlienSignalClassifier()
        self.training_thread = None
        self.initUI()
        
        if self.classifier.load_artifacts():
            self.update_status("ready", "✅ Модель загружена. Готова к работе!")
            self.btn_train.setEnabled(False)
            self.log_message("🎉 Ранее обученная модель загружена автоматически")
        else:
            self.update_status("error", "⚠️  Модель не найдена. Требуется обучение")
    
    def initUI(self):
        # ✅ ИЗМЕНЕНО: Новое название системы
        self.setWindowTitle("🚀 Pypsiki in Space — 2226")
        self.setMinimumSize(900, 700)
        self.setStyleSheet(STYLESHEET)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Заголовок
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setSpacing(5)
        
        # ✅ ИЗМЕНЕНО: Новое название в заголовке
        title = QLabel("🚀 PYPSIKI IN SPACE")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # ✅ ИЗМЕНЕНО: Новый подзаголовок
        subtitle = QLabel("Система классификации инопланетных сигналов")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header)
        
        # Линия
        line = QFrame()
        line.setObjectName("line")
        line.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(line)
        
        # Скролл
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        
        # Статус
        self.status_label = QLabel("Статус: Ожидание...")
        self.status_label.setObjectName("status")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_layout.addWidget(self.status_label)
        
        # Настройки
        config_group = QGroupBox("⚙️ Настройки предобработки (NumPy)")
        config_layout = QVBoxLayout(config_group)
        
        config_grid = QWidget()
        config_grid_layout = QHBoxLayout(config_grid)
        config_grid_layout.setSpacing(20)
        
        self.chk_fft = QCheckBox("📊 FFT-признаки")
        self.chk_stats = QCheckBox("📈 Статистика")
        self.chk_noise = QCheckBox("🔇 Удаление шума")
        self.chk_smooth = QCheckBox("〰️ Сглаживание")
        
        self.chk_stats.setChecked(True)
        self.chk_noise.setChecked(True)
        
        config_grid_layout.addWidget(self.chk_fft)
        config_grid_layout.addWidget(self.chk_stats)
        config_grid_layout.addWidget(self.chk_noise)
        config_grid_layout.addWidget(self.chk_smooth)
        
        config_layout.addWidget(config_grid)
        scroll_layout.addWidget(config_group)
        
        # Обучение
        train_group = QGroupBox("📚 Обучение модели")
        train_layout = QVBoxLayout(train_group)
        
        train_info = QLabel("Загрузите CSV-файл с данными для обучения")
        train_info.setWordWrap(True)
        train_layout.addWidget(train_info)
        
        self.btn_train = QPushButton("📁 Выбрать файл и обучить")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self.start_training)
        train_layout.addWidget(self.btn_train)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        train_layout.addWidget(self.progress_label)
        train_layout.addWidget(self.progress_bar)
        
        scroll_layout.addWidget(train_group)
        
        # Предсказание
        predict_group = QGroupBox("🔍 Классификация сигнала")
        predict_layout = QVBoxLayout(predict_group)
        
        predict_info = QLabel("Введите параметры сигнала (числа через пробел):")
        predict_layout.addWidget(predict_info)
        
        self.entry_signal = QLineEdit()
        self.entry_signal.setPlaceholderText("0.12 0.45 0.78 0.23 0.91 ...")
        predict_layout.addWidget(self.entry_signal)
        
        self.btn_predict = QPushButton("🎯 Классифицировать сигнал")
        self.btn_predict.setObjectName("success")
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_predict.setEnabled(False)
        predict_layout.addWidget(self.btn_predict)
        
        self.result_label = QLabel("📭 Результат: ожидание ввода...")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px; background-color: #0f0f1a; border-radius: 8px;")
        predict_layout.addWidget(self.result_label)
        
        scroll_layout.addWidget(predict_group)
        
        # Лог
        log_group = QGroupBox("📋 Журнал событий")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        self.btn_clear_log = QPushButton("🗑️ Очистить лог")
        self.btn_clear_log.clicked.connect(self.clear_log)
        log_layout.addWidget(self.btn_clear_log)
        
        scroll_layout.addWidget(log_group)
        
        # Подвал
        # ✅ ИЗМЕНЕНО: Новый подвал
        footer = QLabel("© 2226 Pypsiki in Space | Версия 3.0")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #666; padding: 10px;")
        scroll_layout.addWidget(footer)
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        self.log_message("🚀 Pypsiki in Space запущено")
    
    def update_status(self, status_type, message):
        self.status_label.setText(message)
        self.status_label.setObjectName(f"status_{status_type}")
        self.status_label.setStyleSheet(self.status_label.styleSheet())
        self.log_message(f"📢 {message}")
    
    def log_message(self, message):
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def clear_log(self):
        self.log_text.clear()
        self.log_message("🗑️ Лог очищен")
    
    def start_training(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.classifier.config['use_fft'] = self.chk_fft.isChecked()
        self.classifier.config['use_stats'] = self.chk_stats.isChecked()
        self.classifier.config['remove_noise'] = self.chk_noise.isChecked()
        self.classifier.config['smooth_signal'] = self.chk_smooth.isChecked()
        
        self.update_status("training", "⏳ Обучение запущено...")
        self.log_message(f"📂 Загружен файл: {os.path.basename(file_path)}")
        self.log_message(f"⚙️ Настройки: FFT={self.chk_fft.isChecked()}, Статистика={self.chk_stats.isChecked()}")
        
        self.btn_train.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.training_thread = TrainingThread(self.classifier, file_path)
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.log.connect(self.log_message)
        self.training_thread.start()
    
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def on_training_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        
        if success:
            self.update_status("ready", "✅ Обучение завершено!")
            self.btn_predict.setEnabled(True)
            QMessageBox.information(self, "🎉 Успех", message)
            self.log_message(f"✅ {message}")
        else:
            self.update_status("error", "❌ Ошибка обучения!")
            QMessageBox.critical(self, "💥 Ошибка", f"Обучение не удалось:\n\n{message}")
            self.log_message(f"🔥 Ошибка: {message}")
    
    def run_prediction(self):
        if not self.classifier.is_trained:
            QMessageBox.warning(self, "⚠️  Внимание", "Сначала обучите модель!")
            return
        
        signal_str = self.entry_signal.text().strip()
        if not signal_str:
            QMessageBox.warning(self, "⚠️  Внимание", "Введите параметры сигнала!")
            return
        
        try:
            features = [float(x.strip()) for x in signal_str.split() if x.strip()]
            
            name, number, conf = self.classifier.predict_signal(features)
            
            if name:
                result = (
                    f"🛸 <b>Цивилизация:</b> {name}<br>"
                    f"🔢 <b>Код класса:</b> {number}<br>"
                    f"📊 <b>Уверенность:</b> {conf:.2f}%"
                )
                self.result_label.setText(result)
                self.log_message(f"🔍 Предсказание: {name} (класс {number}, {conf:.1f}%)")
                
                if conf > 70:
                    self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px; background-color: #00c853; color: white; border-radius: 8px;")
                elif conf > 40:
                    self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px; background-color: #ff9800; color: white; border-radius: 8px;")
                else:
                    self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px; background-color: #f44336; color: white; border-radius: 8px;")
            else:
                raise ValueError("Не удалось получить предсказание")
                
        except ValueError as e:
            QMessageBox.warning(self, "❌ Ошибка ввода", str(e))
            self.log_message(f"⚠️  Ошибка ввода: {e}")
        except Exception as e:
            QMessageBox.critical(self, "💥 Ошибка", f"Непредвиденная ошибка:\n{str(e)}")
            self.log_message(f"💥 Исключение: {traceback.format_exc()}")


# ==========================================
# ГЕНЕРАЦИЯ ДАННЫХ
# ==========================================
def generate_mock_data(filename='alien_signals.csv'):
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.rand(n_samples, n_features)
    raw_labels = np.random.choice(
        ['Civ_Alpha_X', 'Civ_Beta_9', 'Civ_Gamma_Q', 'Civ_Delta_Z', 'Civ_Epsilon_1'], 
        n_samples
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target_class'] = raw_labels
    df.to_csv(filename, index=False)
    return filename


# ==========================================
# ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    if not os.path.exists('alien_signals.csv'):
        print("🔄 Генерация тестовых данных...")
        generate_mock_data()
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#eee"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#eee"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#eee"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#eee"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#0f3460"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#eee"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#00d9ff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#00d9ff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#00d9ff"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#1a1a2e"))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
