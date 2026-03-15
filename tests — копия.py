"""🧪 Pypsiki in Space — Unit-тесты"""
import pytest, tempfile, numpy as np
from pathlib import Path
from config import DB_PATH
from main import init_db, hash_password, verify_password, authenticate, create_user, get_user, SignalPreprocessor, AlienClassifier

@pytest.fixture(scope="function")
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = Path(f.name)
    init_db(path)
    yield path
    path.unlink()

def test_password_hash():
    p = "test123"; h1, h2 = hash_password(p), hash_password(p)
    assert h1 != h2
    assert verify_password(p, h1) and verify_password(p, h2)
    assert not verify_password("wrong", h1)

def test_auth_success(temp_db):
    pwd = "secure456"; h = hash_password(pwd)
    create_user("u1", h, "Test", "User", "user", db_path=temp_db)
    res = authenticate("u1", pwd, db_path=temp_db)
    assert res and res['username']=="u1" and res['role']=="user" and 'token' in res

def test_auth_fail(temp_db):
    create_user("u2", hash_password("pass"), "A", "B", "user", db_path=temp_db)
    assert authenticate("u2", "wrong", db_path=temp_db) is None
    assert authenticate("nobody", "pass", db_path=temp_db) is None

def test_preprocessor():
    prep = SignalPreprocessor()
    X = np.random.rand(10, 20)
    Xt = prep.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(np.mean(Xt, axis=0), 0, atol=1e-6)
    assert np.allclose(np.std(Xt, axis=0), 1, atol=1e-6)

def test_preprocessor_noise():
    X = np.array([[0.001, 0.5, 0.002, -0.003, 0.8]])
    Xc = SignalPreprocessor.remove_noise(X, thr=0.01)
    assert Xc[0,0]==0 and Xc[0,2]==0 and Xc[0,3]==0
    assert Xc[0,1]==0.5 and Xc[0,4]==0.8

def test_preprocessor_stats():
    X = np.random.rand(5, 50)
    stats = SignalPreprocessor.extract_stats(X)
    assert stats.shape == (5, 11)

def test_classifier_save_load(tmp_path):
    clf = AlienClassifier()
    X = np.random.rand(20, 10); y = np.random.randint(0, 3, 20)
    clf.prep.fit(X); clf.trained = True
    clf.save(tmp_path)
    clf2 = AlienClassifier()
    assert clf2.load(tmp_path)
    assert clf2.trained

def test_classifier_predict():
    clf = AlienClassifier()
    clf.cfg = {'use_fft':False, 'use_stats':False, 'remove_noise':False, 'smooth':False}
    X = np.random.rand(30, 15); y = np.random.randint(0, 2, 30)
    clf.prep.fit(X); clf.trained = True
    from tensorflow import keras
    from tensorflow.keras import layers
    clf.model = keras.Sequential([layers.Input(shape=(15,)), layers.Dense(2, activation='softmax')])
    clf.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    pred, conf, probs = clf.predict(X[:5])
    assert pred.shape == (5,) and conf.shape == (5,) and probs.shape == (5, 2)
    assert np.all((pred >= 0) & (pred < 2))
    assert np.all((conf >= 0) & (conf <= 1))
