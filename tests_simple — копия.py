"""🧪 Pypsiki in Space — Простые тесты (без TensorFlow)"""
import pytest
import tempfile
import numpy as np
from pathlib import Path
from config import DB_PATH

# Импортируем только то, что не зависит от TensorFlow
from main import (
    init_db, hash_password, verify_password, authenticate, 
    create_user, get_user, SignalPreprocessor, get_db
)

@pytest.fixture(scope="function")
def temp_db():
    """Фикстура с временной БД"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = Path(f.name)
    init_db(path)
    yield path
    path.unlink()

def test_password_hash():
    """✅ Тест хеширования пароля"""
    p = "test123"
    h1, h2 = hash_password(p), hash_password(p)
    assert h1 != h2  # разные соли
    assert verify_password(p, h1) and verify_password(p, h2)
    assert not verify_password("wrong", h1)
    print("✅ test_password_hash PASSED")

def test_auth_success(temp_db):
    """✅ Тест успешной аутентификации"""
    pwd = "secure456"
    h = hash_password(pwd)
    create_user("u1", h, "Test", "User", "user", db_path=temp_db)
    res = authenticate("u1", pwd, db_path=temp_db)
    assert res and res['username'] == "u1" and res['role'] == "user" and 'token' in res
    print("✅ test_auth_success PASSED")

def test_auth_fail(temp_db):
    """✅ Тест аутентификации с неверным паролем"""
    create_user("u2", hash_password("pass"), "A", "B", "user", db_path=temp_db)
    assert authenticate("u2", "wrong", db_path=temp_db) is None
    assert authenticate("nobody", "pass", db_path=temp_db) is None
    print("✅ test_auth_fail PASSED")

def test_preprocessor_basic():
    """✅ Тест предобработчика (базовый)"""
    prep = SignalPreprocessor()
    X = np.random.rand(10, 20).astype(np.float32)
    Xt = prep.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(np.mean(Xt, axis=0), 0, atol=1e-5)
    print("✅ test_preprocessor_basic PASSED")

def test_preprocessor_noise():
    """✅ Тест удаления шума"""
    X = np.array([[0.001, 0.5, 0.002, -0.003, 0.8]], dtype=np.float32)
    Xc = SignalPreprocessor.remove_noise(X, thr=0.01)
    assert Xc[0,0] == 0 and Xc[0,2] == 0 and Xc[0,3] == 0
    assert Xc[0,1] == 0.5 and Xc[0,4] == 0.8
    print("✅ test_preprocessor_noise PASSED")

def test_database_persistence(temp_db):
    """✅ Тест сохранения данных в БД"""
    create_user("testuser", hash_password("pwd123"), "Иван", "Иванов", "user", db_path=temp_db)
    user = get_user("testuser", db_path=temp_db)
    assert user is not None
    assert user['first_name'] == "Иван"
    assert user['role'] == "user"
    print("✅ test_database_persistence PASSED")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
