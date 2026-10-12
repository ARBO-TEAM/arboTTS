import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starlette.testclient import TestClient
from arbotts.server import app, get_engine
from arbotts.synthesizer import ArboTTS
import arbotts.server as srv


def test_api_server():
    m_path = REPO_ROOT / "arbotts_v2.onnx"
    c_path = REPO_ROOT / "arbotts_v2.onnx.json"
    if not (m_path.exists() and c_path.exists()):
        m_path = REPO_ROOT / "arbotts_v1.onnx"
        c_path = REPO_ROOT / "arbotts_v1.onnx.json"

    if m_path.exists() and c_path.exists():
        srv._TTS_ENGINE = ArboTTS(model_path=m_path, config_path=c_path)

    client = TestClient(app)

    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Web UI HTML
    r = client.get("/")
    assert r.status_code == 200
    assert "ArboTTS Studio" in r.text
    assert "v2.0" in r.text

    # Speakers list
    r = client.get("/v1/speakers")
    assert r.status_code == 200
    speakers = r.json()
    assert "arbo" in speakers
    assert "arba" in speakers
    assert "sita" in speakers
    assert "maya" in speakers

    # Test speech synthesis endpoints for Arbo & Arba
    if m_path.exists():
        for voice in ["arbo", "arba"]:
            resp = client.post(
                "/v1/audio/speech",
                json={"input": "Halo selamat pagi", "voice": voice, "speed": 0.85},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "audio/wav"
            assert len(resp.content) > 1000


if __name__ == "__main__":
    test_api_server()
    print("PASS: test_api_server")
