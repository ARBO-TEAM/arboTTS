import json
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.models import SynthesizerTrn, MultiPeriodDiscriminator
from training.losses import generator_loss, discriminator_loss


def test_cuda_forward_pass():
    assert torch.cuda.is_available(), "CUDA required for training on RTX 5060 Ti"
    print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")

    with open(REPO_ROOT / "config" / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    net_g = SynthesizerTrn(
        n_vocab=128,
        spec_channels=513,
        segment_size=8192,
        **cfg["model"]
    ).cuda()

    net_d = MultiPeriodDiscriminator().cuda()

    # Create dummy batch
    batch_size = 2
    x = torch.randint(1, 100, (batch_size, 20), device="cuda")
    x_lengths = torch.tensor([20, 15], device="cuda")
    y = torch.randn(batch_size, 513, 80, device="cuda")
    y_lengths = torch.tensor([80, 60], device="cuda")

    # Forward pass Generator
    o, ids_slice, x_mask, y_mask, stats = net_g(x, x_lengths, y, y_lengths)
    assert o.shape[0] == batch_size
    assert o.shape[1] == 1  # 1 audio channel
    print(f"Generator output shape: {o.shape}")

    # Forward pass Discriminator
    y_hat = o
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d(y_hat, y_hat.detach())
    assert len(y_d_rs) > 0
    print(f"Discriminator outputs: {len(y_d_rs)} periods evaluated")

    loss_g, _ = generator_loss(y_d_gs)
    loss_g.backward()
    print(f"Backward pass SUCCESS: loss_g = {loss_g.item():.4f}")


if __name__ == "__main__":
    test_cuda_forward_pass()
    print("ALL CUDA FORWARD & BACKWARD TESTS PASSED ON RTX 5060 Ti!")
