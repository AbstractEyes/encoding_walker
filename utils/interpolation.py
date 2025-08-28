import torch
import torch.nn.functional as F

# --- helper for numerically‑stable slerp ---------------------------------
def slerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor, eps=1e-6) -> torch.Tensor:
    a_norm, b_norm = F.normalize(a, dim=-1, eps=eps), F.normalize(b, dim=-1, eps=eps)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    t1 = torch.sin((1 - alpha) * omega) / sin_omega
    t2 = torch.sin(alpha * omega) / sin_omega
    return t1 * a + t2 * b