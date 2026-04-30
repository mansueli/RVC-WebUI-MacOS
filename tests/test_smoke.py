"""
Narrow smoke tests for launch-critical paths.
All tests run without downloading large model files.
"""
import hashlib
import os
import sys
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Config bootstrap
# ---------------------------------------------------------------------------

def test_config_bootstrap(monkeypatch):
    """Config() must resolve a valid device without crashing on CPU-only CI."""
    monkeypatch.setattr(sys, "argv", ["test", "--nocheck"])
    from configs import Config
    # Reset singleton so each test run gets a fresh instance.
    Config.instance = None
    c = Config()
    assert c.device in ("cpu", "cuda:0", "mps"), f"unexpected device: {c.device}"
    assert isinstance(c.is_half, bool)
    # Reset singleton after test so other tests aren't affected.
    Config.instance = None


# ---------------------------------------------------------------------------
# 2. Audio utilities (pure numpy, no model files needed)
# ---------------------------------------------------------------------------

def test_float_to_int16_range():
    """float_to_int16 must produce values within int16 bounds."""
    from infer.lib.audio import float_to_int16
    rng = np.random.default_rng(42)
    audio = rng.uniform(-1.0, 1.0, 1024).astype(np.float32)
    out = float_to_int16(audio)
    assert out.dtype == np.int16
    assert out.min() >= np.iinfo(np.int16).min
    assert out.max() <= np.iinfo(np.int16).max


def test_float_np_array_to_wav_buf_readable():
    """float_np_array_to_wav_buf must produce a valid WAV that wave can open."""
    from infer.lib.audio import float_np_array_to_wav_buf
    rng = np.random.default_rng(7)
    audio = rng.uniform(-1.0, 1.0, 16000).astype(np.float32)
    buf = float_np_array_to_wav_buf(audio, sr=16000)
    assert isinstance(buf, BytesIO)
    buf.seek(0)
    with wave.open(buf) as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


# ---------------------------------------------------------------------------
# 3. Index path discovery (filesystem walk, no model files needed)
# ---------------------------------------------------------------------------

def test_get_index_path_from_model_found(tmp_path, monkeypatch):
    """get_index_path_from_model must return a path when a matching index exists."""
    idx = tmp_path / "myvoice_IVF256_Flat_nprobe_1_v2.index"
    idx.write_bytes(b"fake")
    monkeypatch.setenv("index_root", str(tmp_path))
    monkeypatch.setenv("outside_index_root", str(tmp_path))
    # Reset any cached fairseq import state by importing fresh
    import importlib, infer.modules.vc.utils as utils_mod
    importlib.reload(utils_mod)
    result = utils_mod.get_index_path_from_model("myvoice.pth")
    assert result != "", "expected a matching index path to be found"
    assert "myvoice" in result


def test_get_index_path_from_model_not_found(tmp_path, monkeypatch):
    """get_index_path_from_model must return empty string when no index matches."""
    monkeypatch.setenv("index_root", str(tmp_path))
    monkeypatch.setenv("outside_index_root", str(tmp_path))
    import importlib, infer.modules.vc.utils as utils_mod
    importlib.reload(utils_mod)
    result = utils_mod.get_index_path_from_model("nonexistent.pth")
    assert result == ""


# ---------------------------------------------------------------------------
# 4. Asset hash check (rvcmd.check_model with a real temp file)
# ---------------------------------------------------------------------------

def test_check_model_valid(tmp_path, monkeypatch):
    """check_model must return True when file hash matches the expected value."""
    data = b"hello rvc"
    digest = hashlib.sha256(data).hexdigest()
    f = tmp_path / "test.pt"
    f.write_bytes(data)
    monkeypatch.setenv("sha256_test_pt", digest)

    from infer.lib.rvcmd import check_model
    assert check_model(tmp_path, "test.pt", digest) is True


def test_check_model_missing(tmp_path):
    """check_model must return False when the file does not exist."""
    from infer.lib.rvcmd import check_model
    assert check_model(tmp_path, "missing.pt", "abc123") is False


def test_check_model_hash_mismatch(tmp_path):
    """check_model must return False when the file hash does not match."""
    f = tmp_path / "corrupt.pt"
    f.write_bytes(b"wrong content")
    from infer.lib.rvcmd import check_model
    assert check_model(tmp_path, "corrupt.pt", "0" * 64) is False
