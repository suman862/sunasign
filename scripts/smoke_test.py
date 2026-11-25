#testing 
import torch
from suna_sign.model import SunaSign

cfg = {
    "image_dim": 64, "landmark_dim": 32, "d_model": 64,
    "nhead": 4, "layers": 2, "ff_dim": 128, "dropout": 0.1,
    "num_sign_classes": 10, "vocab_size": 50, "max_text_len": 16,
}
m = SunaSign(cfg)
frames = torch.randn(2, 8, 3, 64, 64)
lm = torch.randn(2, 8, 32)
tok = torch.randint(0, 50, (2, 6))
pad = torch.zeros(2, 8, dtype=torch.bool)
out = m(frames, lm, tok, pad)
assert out.sign_logits.shape == (2,8,10)
assert out.boundary_logits.shape == (2,8)
assert out.text_logits.shape == (2,6,50)
print("SMOKE TEST PASSED")
