
#loss 
import torch
import torch.nn.functional as F


def compute_loss(out, tokens, boundary, padding_mask, pad_id=0,
                 w_sign=1.0, w_boundary=0.25, w_text=1.0):
    valid = ~padding_mask
    sign_target = torch.zeros_like(out.sign_logits[..., 0], dtype=torch.long)
    # If sign labels are added to the dataset, replace this target with them.
    # Keeping it disabled avoids silently training a meaningless sign classifier.
    sign_loss = out.sign_logits.sum() * 0.0

    b_loss = F.binary_cross_entropy_with_logits(
        out.boundary_logits[valid], boundary[valid]
    )

    if tokens.size(1) < 2:
        raise ValueError("token sequence must contain BOS/EOS")
    text_logits = out.text_logits[:, :-1].reshape(-1, out.text_logits.size(-1))
    text_targets = tokens[:, 1:].reshape(-1)
    text_loss = F.cross_entropy(text_logits, text_targets, ignore_index=pad_id)

    return w_sign*sign_loss + w_boundary*b_loss + w_text*text_loss, {
        "sign": float(sign_loss.detach()),
        "boundary": float(b_loss.detach()),
        "text": float(text_loss.detach()),
    }
