import json
from pathlib import Path
from tools.create_tier_config import generate_tier_config
from arbotts.downloader import get_model_path, TIER_FILES

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_tiers_config_validity():
    tiers_path = REPO_ROOT / "configs" / "tiers.json"
    assert tiers_path.exists(), "configs/tiers.json must exist"

    with open(tiers_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiers = data.get("tiers", {})
    assert "standard" in tiers
    assert "small" in tiers
    assert "tiny" in tiers

    assert tiers["standard"]["parameters_m"] > tiers["small"]["parameters_m"]
    assert tiers["small"]["parameters_m"] > tiers["tiny"]["parameters_m"]


def test_generate_tier_configs(tmp_path):
    for tier in ["small", "tiny"]:
        out_cfg = tmp_path / f"arbotts_{tier}.onnx.json"
        cfg = generate_tier_config(tier, out_path=out_cfg)
        assert out_cfg.exists()
        assert cfg["tier"] == tier
        assert "model" in cfg
        assert cfg["model"]["hidden_channels"] in [128, 96]


def test_downloader_tier_resolution():
    m_std, c_std = get_model_path(tier="standard")
    assert m_std.exists()
    assert c_std.exists()

    # If small/tiny model not yet on disk, it safely falls back to standard
    m_sm, c_sm = get_model_path(tier="small")
    assert m_sm.exists()
    assert c_sm.exists()
