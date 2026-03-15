"""🎨 Pypsiki in Space — Графический интерфейс (PyQt6)"""
import sys, numpy as np, config, traceback
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from main import authenticate, DataNPZLoader, AlienClassifier, plot_accuracy, plot_distribution, plot_predictions, plot_top, init_db

# ===== СТИЛИ =====
STYLES = f"""
QMainWindow, QWidget {{ background-color: {config.COLORS['bg_primary']}; color: {config.COLORS['text_primary']}; font-family: 'Segoe UI', Arial; }}
QLabel#title {{ font-size: 28px; font-weight: bold; color: {config.COLORS['accent']}; padding: 15px; }}
QLabel#subtitle {{ font-size: 14px; color: {config.COLORS['text_secondary']}; }}
QGroupBox {{ font-weight: bold; color: {config.COLORS['accent']}; border: 2px solid {config.COLORS['bg_secondary']}; border-radius: 12px; margin-top: 20px; padding-top: 20px; background-color: {config.COLORS['bg_secondary']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 20px; padding: 0 15px; }}
QPushButton {{ background-color: {config.COLORS['bg_tertiary']}; color: white; border: 2px solid {config.COLORS['bg_secondary']}; border-radius: 10px; padding: 12px 30px; font-size: 14px; font-weight: 600; min-width: 120px; }}
QPushButton:hover {{ background-color: {config.COLORS['bg_secondary']}; border-color: {config.COLORS['accent']}; }}
QPushButton:pressed {{ background-color: {config.COLORS['accent']}; color: {config.COLORS['bg_primary']}; }}
QPushButton#primary {{ background-color: {config.COLORS['accent']}; color: {config.COLORS['bg_primary']}; border: none; }}
QPushButton#primary:hover {{ background-color: {config.COLORS['accent_hover']}; }}
QPushButton#success {{ background-color: {config.COLORS['success']}; border: none; }}
QPushButton#danger {{ background-color: {config.COLORS['error']}; border: none; }}
QPushButton#danger:hover {{ background-color: #d32f2f; }}
QLineEdit, QComboBox {{ background-color: {config.COLORS['bg_tertiary']}; border: 2px solid {config.COLORS['bg_secondary']}; border-radius: 8px; padding: 10px 15px; color: white; }}
QLineEdit:focus {{ border-color: {config.COLORS['accent']}; }}
QTextEdit {{ background-color: {config.COLORS['bg_tertiary']}; border: 2px solid {config.COLORS['bg_secondary']}; border-radius: 8px; padding: 10px; font-family: Consolas; color: {config.COLORS['accent']}; }}
QCheckBox {{ color: white; spacing: 10px; }}
QCheckBox::indicator {{ width: 22px; height: 22px; border-radius: 6px; border: 2px solid {config.COLORS['bg_secondary']}; background-color: {config.COLORS['bg_tertiary']}; }}
QCheckBox::indicator:checked {{ background-color: {config.COLORS['accent']}; border-color: {config.COLORS['accent']}; }}
QProgressBar {{ background-color: {config.COLORS['bg_tertiary']}; border: 2px solid {config.COLORS['bg_secondary']}; border-radius: 10px; height: 24px; text-align: center; color: white; font-weight: bold; }}
QProgressBar::chunk {{ background-color: {config.COLORS['accent']}; border-radius: 8px; }}
QLabel#status {{ padding: 12px; border-radius: 10px; font-weight: 600; }}
QLabel#status_ready {{ background-color: {config.COLORS['success']}; color: white; }}
QLabel#status_training {{ background-color: {config.COLORS['warning']}; color: white; }}
QLabel#status_error {{ background-color: {config.COLORS['error']}; color: white; }}
QFrame#line {{ background-color: {config.COLORS['bg_secondary']}; max-height: 2px; }}
QScrollBar:vertical {{ background-color: {config.COLORS['bg_secondary']}; width: 14px; border-radius: 7px; }}
QScrollBar::handle:vertical {{ background-color: {config.COLORS['accent']}; border-radius: 7px; min-height: 30px; }}
QToolTip {{ background-color: {config.COLORS['bg_tertiary']}; color: white; border: 1px solid {config.COLORS['accent']}; border-radius: 5px; padding: 8px; }}
QTableWidget {{ background-color: {config.COLORS['bg_tertiary']}; color: white; gridline-color: {config.COLORS['bg_secondary']}; }}
QTableWidget::item {{ color: white; }}
QHeaderView::section {{ background-color: {config.COLORS['bg_secondary']}; color: white; padding: 8px; border: none; }}
"""

# ===== ПОТОК ОБУЧЕНИЯ =====
class TrainThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, dict)
    def __init__(self, clf, Xtr, ytr, Xv, yv):
        super().__init__()
        self.clf, self.Xtr, self.ytr, self.Xv, self.yv = clf, Xtr, ytr, Xv, yv
    def run(self):
        try:
            def on_ep(logs): 
                try: self.progress.emit(int(logs['epoch']/config.EPOCHS*100), f"Эпоха {logs['epoch']}/{config.EPOCHS}")
                except: pass
            hist = self.clf.train(self.Xtr, self.ytr, self.Xv if len(self.Xv)>0 else None, self.yv if len(self.yv)>0 else None, on_epoch=on_ep)
            self.finished.emit(True, f"✅ Точность: {hist['accuracy'][-1]*100:.1f}%", hist)
        except Exception as e: 
            self.finished.emit(False, str(e), {})

# ===== CANVAS ДЛЯ ГРАФИКОВ (БЕЛЫЙ ТЕКСТ) =====
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, w=10, h=4):
        self.fig = plt.figure(figsize=(w,h), dpi=100, facecolor='#16213e')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1a1a2e')
        # ✅ Белый цвет для всех элементов графика
        for s in self.ax.spines.values(): s.set_color('white')
        self.ax.tick_params(colors='white')
        self.ax.set_xlabel('Эпоха', color='white')
        self.ax.set_ylabel('Значение', color='white')
        self.ax.title.set_color('white')
        for label in self.ax.get_xticklabels() + self.ax.get_yticklabels():
            label.set_color('white')
        super().__init__(self.fig)
        self.setParent(parent)

# ===== ОКНО ВХОДА =====
class LoginWindow(QWidget):
    logged_in = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle(f"{config.APP_NAME} — Вход")
        self.setMinimumSize(450, 350)
        self.setStyleSheet(STYLES)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40,40,40,40)
        
        title = QLabel("🚀 PYPSIKI IN SPACE")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        
        subtitle = QLabel("Система классификации инопланетных сигналов")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(subtitle)
        
        form = QWidget()
        fl = QFormLayout(form)
        self.user = QLineEdit()
        self.user.setPlaceholderText("Логин")
        self.pwd = QLineEdit()
        self.pwd.setPlaceholderText("Пароль")
        self.pwd.setEchoMode(QLineEdit.EchoMode.Password)
        self.user.setFixedHeight(45)
        self.pwd.setFixedHeight(45)
        self.pwd.returnPressed.connect(self.login)
        fl.addRow("Логин:", self.user)
        fl.addRow("Пароль:", self.pwd)
        lay.addWidget(form)
        
        btn = QPushButton("🔓 Войти")
        btn.setObjectName("primary")
        btn.setFixedHeight(50)
        btn.clicked.connect(self.login)
        lay.addWidget(btn)
        
        info = QLabel("💡 По умолчанию: admin / admin123")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"color: {config.COLORS['text_secondary']};")
        lay.addWidget(info)
        
        lay.addStretch()
        self.user.setFocus()
    
    def login(self):
        try:
            u, p = self.user.text().strip(), self.pwd.text()
            if not u or not p: QMessageBox.warning(self, "⚠️", "Заполните все поля!"); return
            user = authenticate(u, p)
            if user: 
                self.logged_in.emit(user)
                self.hide()
            else: 
                QMessageBox.critical(self, "❌", "Неверный логин или пароль!\nПопробуйте: admin / admin123")
                self.pwd.clear()
                self.pwd.setFocus()
        except Exception as e:
            QMessageBox.critical(self, "❌", f"Ошибка входа: {str(e)}")

# ===== ДАШБОРД =====
class Dashboard(QWidget):
    def __init__(self, user):
        super().__init__()
        self.user, self.clf, self.data = user, AlienClassifier(), None
        self.initUI()
        if self.clf.load(): self.set_status("✅ Модель загружена", "ready")
        else: self.set_status("⚠️ Модель не найдена — обучите", "error")
    def initUI(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20,20,20,20)
        self.status = QLabel("Статус: ...")
        self.status.setObjectName("status")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("padding:10px;border-radius:8px;font-weight:bold;")
        lay.addWidget(self.status)
        ctrl = QGroupBox("⚙️ Управление")
        cl = QHBoxLayout(ctrl)
        self.btn_load = QPushButton("📁 Загрузить .npz")
        self.btn_load.clicked.connect(self.load_data)
        cl.addWidget(self.btn_load)
        self.btn_train = QPushButton("🎓 Обучить")
        self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self.train)
        cl.addWidget(self.btn_train)
        self.btn_pred = QPushButton("🎯 Классифицировать")
        self.btn_pred.setObjectName("success")
        self.btn_pred.clicked.connect(self.predict)
        self.btn_pred.setEnabled(False)
        cl.addWidget(self.btn_pred)
        if self.user['role']=='admin':
            self.btn_admin = QPushButton("👥 Пользователи")
            self.btn_admin.clicked.connect(lambda: self.parent().parent().show_admin() if hasattr(self.parent().parent(),'show_admin') else None)
            cl.addWidget(self.btn_admin)
        lay.addWidget(ctrl)
        self.prog = QProgressBar()
        self.prog.setVisible(False)
        self.prog_lbl = QLabel("")
        self.prog_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.prog_lbl)
        lay.addWidget(self.prog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sc = QWidget()
        scl = QVBoxLayout(sc)
        scl.setSpacing(20)
        self.g1 = QGroupBox("📈 Точность обучения")
        self.c1 = MplCanvas(self.g1)
        self.g1.setLayout(QVBoxLayout())
        self.g1.layout().addWidget(self.c1)
        scl.addWidget(self.g1)
        self.g2 = QGroupBox("🥧 Распределение классов")
        self.c2 = MplCanvas(self.g2)
        self.g2.setLayout(QVBoxLayout())
        self.g2.layout().addWidget(self.c2)
        scl.addWidget(self.g2)
        self.g3 = QGroupBox("🎯 Точность предсказаний")
        self.c3 = MplCanvas(self.g3)
        self.g3.setLayout(QVBoxLayout())
        self.g3.layout().addWidget(self.c3)
        scl.addWidget(self.g3)
        self.g4 = QGroupBox("🏆 Топ-5 классов")
        self.c4 = MplCanvas(self.g4)
        self.g4.setLayout(QVBoxLayout())
        self.g4.layout().addWidget(self.c4)
        scl.addWidget(self.g4)
        self.g5 = QGroupBox("📋 Результат")
        self.res = QLabel("📭 Введите сигнал для классификации")
        self.res.setWordWrap(True)
        self.res.setStyleSheet("font-size:14px;padding:15px;background-color:#0f0f1a;border-radius:8px;")
        self.g5.setLayout(QVBoxLayout())
        self.g5.layout().addWidget(self.res)
        scl.addWidget(self.g5)
        scroll.setWidget(sc)
        lay.addWidget(scroll)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("line")
        lay.addWidget(line)
        il = QHBoxLayout()
        il.addWidget(QLabel("📡 Сигнал:"))
        self.inp = QLineEdit()
        self.inp.setPlaceholderText("0.12 0.45 0.78 ...")
        il.addWidget(self.inp)
        lay.addLayout(il)
    def set_status(self, msg, st):
        self.status.setText(msg)
        self.status.setObjectName(f"status_{st}")
        self.status.setStyleSheet(self.status.styleSheet())
    def load_data(self):
        try:
            fp, _ = QFileDialog.getOpenFileName(self, "Данные", "", "NPZ (*.npz)")
            if not fp: return
            self.set_status("⏳ Загрузка данных...", "training")
            self.prog.setVisible(True)
            self.prog.setValue(50)
            self.prog_lbl.setText("Обработка данных...")
            
            loader = DataNPZLoader(local_path=fp)
            data = loader.load()
            Xtr, ytr, Xv, yv, enc = loader.prepare(data)
            
            self.prog.setValue(75)
            self.prog_lbl.setText("Построение графиков...")
            
            self.data = {'Xtr':Xtr,'ytr':ytr,'Xv':Xv,'yv':yv,'enc':enc}
            self.update_chart(self.c2, lambda: plot_distribution(ytr))
            if len(yv)>0: self.update_chart(self.c4, lambda: plot_top(yv))
            
            self.prog.setValue(100)
            self.prog.setVisible(False)
            self.set_status(f"✅ Загружено: {len(ytr)} train, {len(yv)} valid", "ready")
        except Exception as e:
            self.prog.setVisible(False)
            QMessageBox.critical(self, "❌", f"Ошибка: {str(e)}")
    def train(self):
        try:
            if not self.data or 'Xtr' not in self.data:
                QMessageBox.warning(self, "⚠️", "Сначала загрузите данные!")
                return
            self.prog.setVisible(True)
            self.prog.setValue(0)
            self.set_status("⏳ Обучение...", "training")
            Xtr,ytr,Xv,yv = self.data['Xtr'],self.data['ytr'],self.data['Xv'],self.data['yv']
            self.thread = TrainThread(self.clf, Xtr, ytr, Xv, yv)
            self.thread.progress.connect(lambda v,m: [self.prog.setValue(v), self.prog_lbl.setText(m)])
            def on_done(ok, msg, hist):
                try:
                    self.prog.setVisible(False)
                    if ok:
                        try: self.clf.save()
                        except: pass
                        if hist and 'accuracy' in hist:
                            try: self.update_chart(self.c1, lambda: plot_accuracy(hist))
                            except: pass
                        if len(Xv)>0 and len(yv)>0:
                            try:
                                pred,conf,_ = self.clf.predict(Xv)
                                self.update_chart(self.c3, lambda: plot_predictions(yv, pred, conf))
                                res = self.clf.evaluate(Xv, yv)
                                self.set_status(f"✅ Готово! Точность: {res['accuracy']*100:.1f}%", "ready")
                            except: self.set_status(f"✅ Готово! {msg}", "ready")
                        else: self.set_status(f"✅ Готово! {msg}", "ready")
                        self.btn_pred.setEnabled(True)
                        QMessageBox.information(self, "🎉", msg)
                    else:
                        self.set_status("❌ Ошибка", "error")
                        QMessageBox.critical(self, "💥", f"Не удалось: {msg}")
                except Exception as e:
                    self.set_status("❌ Ошибка", "error")
                    QMessageBox.critical(self, "💥", f"Ошибка: {str(e)}")
            self.thread.finished.connect(on_done)
            self.thread.start()
        except Exception as e:
            self.prog.setVisible(False)
            self.set_status("❌ Ошибка", "error")
            QMessageBox.critical(self, "💥", f"Ошибка обучения: {str(e)}")
    def predict(self):
        try:
            if not self.clf.trained: QMessageBox.warning(self, "⚠️", "Обучите модель!"); return
            s = self.inp.text().strip()
            if not s: QMessageBox.warning(self, "⚠️", "Введите сигнал!"); return
            feat = np.array([float(x) for x in s.split()])
            pred, conf, _ = self.clf.predict(feat.reshape(1,-1))
            name = self.data['enc'].inverse_transform([pred[0]])[0] if self.data and 'enc' in self.data else f"Class {pred[0]}"
            self.res.setText(f"🛸 <b>Цивилизация:</b> {name}<br>🔢 <b>Код:</b> {pred[0]+1}<br>📊 <b>Уверенность:</b> {conf[0]*100:.2f}%")
            if conf[0]>0.7: self.res.setStyleSheet("font-size:14px;padding:15px;background-color:#00c853;color:white;border-radius:8px;")
            elif conf[0]>0.4: self.res.setStyleSheet("font-size:14px;padding:15px;background-color:#ff9800;color:white;border-radius:8px;")
            else: self.res.setStyleSheet("font-size:14px;padding:15px;background-color:#f44336;color:white;border-radius:8px;")
        except Exception as e: QMessageBox.warning(self, "❌", f"Ошибка: {str(e)}")
    def update_chart(self, canvas, plot_fn):
        try:
            canvas.ax.clear()
            fig = plot_fn()
            canvas.figure = fig
            fig.canvas = canvas
            canvas.draw()
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Ошибка графика: {e}")

# ===== АДМИН ПАНЕЛЬ =====
class AdminPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_users()
    def initUI(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20,20,20,20)
        lbl = QLabel("👥 Управление пользователями")
        lbl.setStyleSheet("font-size:16px;font-weight:bold;color:#00d9ff;")
        lay.addWidget(lbl)
        form = QWidget()
        fl = QFormLayout(form)
        self.u = QLineEdit()
        self.p = QLineEdit()
        self.p.setEchoMode(QLineEdit.EchoMode.Password)
        self.fn = QLineEdit()
        self.ln = QLineEdit()
        self.role = QComboBox()
        self.role.addItems(["user","admin"])
        for w in [self.u,self.p,self.fn,self.ln]: w.setPlaceholderText("...")
        fl.addRow("Логин *:",self.u)
        fl.addRow("Пароль *:",self.p)
        fl.addRow("Имя *:",self.fn)
        fl.addRow("Фамилия *:",self.ln)
        fl.addRow("Роль:",self.role)
        lay.addWidget(form)
        btn_layout = QHBoxLayout()
        btn = QPushButton("✨ Создать")
        btn.setObjectName("success")
        btn.clicked.connect(self.create)
        btn_layout.addWidget(btn)
        btn_delete = QPushButton("🗑️ Удалить")
        btn_delete.setObjectName("danger")
        btn_delete.clicked.connect(self.delete_user)
        btn_layout.addWidget(btn_delete)
        lay.addLayout(btn_layout)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID","Логин","Имя","Фамилия","Роль"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        lay.addWidget(QLabel("📋 Пользователи:"))
        lay.addWidget(self.table)
        lay.addStretch()
    def load_users(self):
        try:
            from main import get_db
            with get_db() as conn:
                users = conn.cursor().execute("SELECT id,username,first_name,last_name,role FROM users").fetchall()
                self.table.setRowCount(len(users))
                for r,u in enumerate(users):
                    for c,v in enumerate(u):
                        it = QTableWidgetItem(str(v))
                        it.setForeground(QColor("white"))  # ✅ Белый текст
                        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.table.setItem(r,c,it)
        except Exception as e:
            QMessageBox.critical(self, "❌", f"Ошибка загрузки: {str(e)}")
    def create(self):
        try:
            from main import create_user, hash_password
            u,p,fn,ln,rl = self.u.text().strip(),self.p.text(),self.fn.text().strip(),self.ln.text().strip(),self.role.currentText()
            if not all([u,p,fn,ln]): QMessageBox.warning(self,"⚠️","Заполните все поля!"); return
            if len(p)<6: QMessageBox.warning(self,"⚠️","Пароль ≥6 символов!"); return
            create_user(u, hash_password(p), fn, ln, rl)
            QMessageBox.information(self,"✅",f"Пользователь '{u}' создан!")
            self.u.clear()
            self.p.clear()
            self.fn.clear()
            self.ln.clear()
            self.role.setCurrentIndex(0)
            self.load_users()
        except Exception as e: QMessageBox.critical(self,"❌",f"Ошибка: {str(e)}")
    def delete_user(self):
        try:
            selected = self.table.selectedItems()
            if not selected:
                QMessageBox.warning(self, "⚠️", "Выберите пользователя для удаления!")
                return
            row = selected[0].row()
            user_id = self.table.item(row, 0).text()
            username = self.table.item(row, 1).text()
            if username == 'admin':
                QMessageBox.warning(self, "⚠️", "Нельзя удалить главного администратора!")
                return
            confirm = QMessageBox.question(self, "❓ Подтверждение", 
                                          f"Удалить пользователя '{username}'?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if confirm == QMessageBox.StandardButton.Yes:
                from main import get_db
                with get_db() as conn:
                    conn.cursor().execute("DELETE FROM users WHERE id = ?", (user_id,))
                QMessageBox.information(self, "✅", f"Пользователь '{username}' удалён!")
                self.load_users()
        except Exception as e: 
            QMessageBox.critical(self,"❌",f"Ошибка: {str(e)}")

# ===== ГЛАВНОЕ ОКНО =====
class MainWindow(QMainWindow):
    logout_requested = pyqtSignal()
    def __init__(self, user):
        super().__init__()
        self.user, self.stack = user, QStackedWidget()
        self.initUI()
    def initUI(self):
        self.setWindowTitle(config.APP_NAME)
        self.setMinimumSize(1100,750)
        self.setStyleSheet(STYLES)
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)
        lay.setSpacing(0)
        lay.setContentsMargins(0,0,0,0)
        header = QWidget()
        header.setStyleSheet(f"background-color:{config.COLORS['bg_secondary']};padding:15px 30px;")
        hl = QHBoxLayout(header)
        logo = QLabel("🚀 Pypsiki in Space")
        logo.setStyleSheet("font-size:20px;font-weight:bold;color:#00d9ff;")
        hl.addWidget(logo)
        hl.addStretch()
        info = QLabel(f"👤 {self.user['first_name']} {self.user['last_name']} ({self.user['role']})")
        info.setStyleSheet("color:#888;")
        hl.addWidget(info)
        logout = QPushButton("🚪 Выход")
        logout.setFixedWidth(100)
        logout.clicked.connect(lambda: self.logout_requested.emit())
        hl.addWidget(logout)
        lay.addWidget(header)
        lay.addWidget(self.stack)
        self.dash = Dashboard(self.user)
        self.stack.addWidget(self.dash)
        if self.user['role']=='admin':
            self.admin = AdminPanel()
            self.stack.addWidget(self.admin)
            self.dash.btn_admin.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        # ✅ ИСПРАВЛЕННЫЙ ПОДВАЛ
        footer = QLabel("Pypsiki in Space | 2026")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(f"color:{config.COLORS['text_secondary']};padding:10px;background-color:{config.COLORS['bg_secondary']};")
        lay.addWidget(footer)
    def show_admin(self): self.stack.setCurrentIndex(1)

# ===== ЗАПУСК =====
def run_app():
    try:
        init_db()
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(config.COLORS['bg_primary']))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(config.COLORS['text_primary']))
        pal.setColor(QPalette.ColorRole.Base, QColor(config.COLORS['bg_tertiary']))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(config.COLORS['bg_secondary']))
        pal.setColor(QPalette.ColorRole.Text, QColor(config.COLORS['text_primary']))
        pal.setColor(QPalette.ColorRole.Button, QColor(config.COLORS['bg_secondary']))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(config.COLORS['text_primary']))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(config.COLORS['accent']))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(config.COLORS['bg_primary']))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(config.COLORS['bg_primary']))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(config.COLORS['text_primary']))
        pal.setColor(QPalette.ColorRole.Link, QColor(config.COLORS['accent']))
        pal.setColor(QPalette.ColorRole.BrightText, QColor(config.COLORS['accent']))
        app.setPalette(pal)
        login = LoginWindow()
        main_window_ref = []
        def show_main(user):
            win = MainWindow(user)
            main_window_ref.append(win)
            def on_logout():
                win.hide()
                login.user.setText('')
                login.pwd.setText('')
                login.user.setFocus()
                login.show()
            win.logout_requested.connect(on_logout)
            win.show()
            login.hide()
        login.logged_in.connect(show_main)
        login.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Критическая ошибка запуска: {e}")
        traceback.print_exc()
