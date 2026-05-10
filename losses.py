"""
Knowledge Distillation Loss Functions.

Three loss components, combined with weights α and β:

    L_total = (1-α) * L_CE(student, y_true)          ← hard-label loss
            +    α  * L_KD(student, teacher, T)      ← soft-label loss
            +    β  * L_feat(student_feat, teacher_feat)  ← feature loss

Each is defined below with full mathematical detail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────
# 1. SOFT-LABEL DISTILLATION (Hinton et al., 2015)
# ─────────────────────────────────────────────────────────────────────
class SoftLabelKDLoss(nn.Module):
    """
    KL divergence between temperature-softened teacher and student outputs.

    Motivation — "dark knowledge":
        A trained teacher doesn't just know "this is a cat". It knows
        "this is mostly a cat (0.85), a bit like a tiger (0.10), a bit
        like a dog (0.03), definitely not a truck (1e-9)". The *relative*
        probabilities of the wrong classes encode how the teacher sees
        the similarity structure of the world. That's dark knowledge.

        At default softmax (T=1), this is mostly hidden because the
        correct class dominates. Dividing logits by T > 1 SOFTENS the
        distribution, spreading mass across classes and making the
        dark knowledge visible.

    Math:
        p_T = softmax(z_T / T)         ← teacher soft targets
        p_S = softmax(z_S / T)         ← student soft outputs
        L_KD = T² * KL(p_T || p_S)

        The T² factor (Hinton) compensates for the fact that gradients
        from the softened softmax are scaled by 1/T², so without this
        multiplier the KD loss contribution would shrink as T grows,
        and you'd have to re-tune α every time you changed T.

    PyTorch gotcha:
        nn.KLDivLoss expects its INPUT (student) as LOG-probs and its
        TARGET (teacher) as regular probs. Getting this backward is a
        silent bug — the loss value still looks reasonable but trains
        badly. We handle it explicitly.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,  # [B, num_classes]
        teacher_logits: torch.Tensor,  # [B, num_classes]
    ) -> torch.Tensor:
        # Student as log-probs (input to KLDivLoss)
        log_p_student = F.log_softmax(student_logits / self.T, dim=-1)
        # Teacher as probs (target for KLDivLoss)
        p_teacher = F.softmax(teacher_logits / self.T, dim=-1)

        # batchmean → sum over classes, mean over batch (correct KL form)
        kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

        # T² rescaling
        return kl * (self.T ** 2)


# ─────────────────────────────────────────────────────────────────────
# 2. FEATURE-BASED DISTILLATION (FitNets, Romero et al., 2015)
# ─────────────────────────────────────────────────────────────────────
class FeatureKDLoss(nn.Module):
    """
    MSE between teacher and student intermediate feature maps.

    Motivation:
        Matching only final outputs (soft labels) is a shallow signal —
        the student is free to reach the same answer via totally
        different internal representations. Forcing intermediate
        features to match gives a much richer training signal: the
        student must develop teacher-like feature extractors.

    Math:
        For each stage s, with teacher features F^T_s and student features F^S_s:
            L_feat_s = MSE( normalize(adapt(F^S_s)), normalize(F^T_s) )
        L_feat = mean over stages of L_feat_s

    Adaptation layers:
        If student and teacher have different channel dimensions at
        stage s, we can't directly compare their features. We project
        the student to the teacher's channel count with a 1×1 conv:
            F^S_adapted = Conv1x1(F^S) — no activation, no BN
        Here teacher and student share channel dims per stage (both
        use CIFAR ResNet with base_channels=16), so the adaptation
        layers reduce to identity — but we build the general case so
        the code transfers to any teacher/student pair.

    L2-normalization:
        Before MSE, we L2-normalize features along the channel axis.
        This makes the loss scale-invariant: we're matching *direction*
        of the feature vector at each spatial location, not its
        magnitude. Without this, a student with smaller weight norms
        would have a systematic disadvantage.

    Detaching the teacher:
        We .detach() teacher features before computing MSE. Critical!
        Without this, gradients would flow back into the teacher and
        PyTorch would either (a) update it, (b) error because it's in
        eval mode, or (c) silently waste memory. Detach is the fix.
    """

    def __init__(self, teacher_channels: List[int], student_channels: List[int]):
        super().__init__()
        assert len(teacher_channels) == len(student_channels), (
            "Teacher and student must expose the same number of feature stages"
        )

        # One adaptation layer per stage
        self.adaptation_layers = nn.ModuleList()
        for t_ch, s_ch in zip(teacher_channels, student_channels):
            if t_ch == s_ch:
                self.adaptation_layers.append(nn.Identity())
            else:
                # 1x1 conv projection (+ BN for stability)
                self.adaptation_layers.append(
                    nn.Sequential(
                        nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(t_ch),
                    )
                )

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
    ) -> torch.Tensor:
        total = 0.0
        n = len(student_features)

        for s_feat, t_feat, adapt in zip(
            student_features, teacher_features, self.adaptation_layers
        ):
            # Project student to teacher channel dim (identity if matched)
            s_adapted = adapt(s_feat)

            # Spatially align if needed (usually matches for same-arch pair)
            if s_adapted.shape[2:] != t_feat.shape[2:]:
                s_adapted = F.interpolate(
                    s_adapted, size=t_feat.shape[2:], mode="bilinear", align_corners=False
                )

            # L2-normalize along channel axis → direction matching
            s_norm = F.normalize(s_adapted, p=2, dim=1)
            t_norm = F.normalize(t_feat.detach(), p=2, dim=1)  # DETACH!

            total = total + F.mse_loss(s_norm, t_norm)

        return total / n


# ─────────────────────────────────────────────────────────────────────
# 3. COMBINED DISTILLATION LOSS
# ─────────────────────────────────────────────────────────────────────
class DistillationLoss(nn.Module):
    """
    The full distillation objective, combining all three components.

        L = (1-α) * L_CE + α * L_KD + β * L_feat

    Args:
        temperature: softmax temperature T
        alpha: weight on soft-label KD (0 → pure hard-label training,
               1 → pure KD, 0.5 → balanced)
        beta: weight on feature distillation
        teacher_channels: per-stage channel dims of the teacher
        student_channels: per-stage channel dims of the student

    Set beta=0 to disable feature distillation (logit-only KD).
    Set alpha=0 AND beta=0 to recover pure cross-entropy training.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        beta: float = 100.0,
        teacher_channels: List[int] = None,
        student_channels: List[int] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.ce_loss = nn.CrossEntropyLoss()
        self.kd_loss = SoftLabelKDLoss(temperature=temperature)

        self.feature_loss = None
        if beta > 0 and teacher_channels is not None and student_channels is not None:
            self.feature_loss = FeatureKDLoss(
                teacher_channels=teacher_channels,
                student_channels=student_channels,
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: List[torch.Tensor] = None,
        teacher_features: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, dict of component values for logging).
        """
        # Hard-label cross-entropy
        l_ce = self.ce_loss(student_logits, labels)

        # Soft-label KD
        l_kd = self.kd_loss(student_logits, teacher_logits)

        # Feature KD (optional)
        l_feat = torch.tensor(0.0, device=student_logits.device)
        if self.feature_loss is not None and student_features is not None:
            l_feat = self.feature_loss(student_features, teacher_features)

        total = (1 - self.alpha) * l_ce + self.alpha * l_kd + self.beta * l_feat

        return total, {
            "total": total.item(),
            "ce": l_ce.item(),
            "kd": l_kd.item(),
            "feat": l_feat.item() if isinstance(l_feat, torch.Tensor) else l_feat,
        }


if __name__ == "__main__":
    # Smoke test
    B, C = 8, 100
    s_logits = torch.randn(B, C, requires_grad=True)
    t_logits = torch.randn(B, C)
    labels = torch.randint(0, C, (B,))

    # Fake feature maps
    s_feats = [torch.randn(B, 16, 32, 32, requires_grad=True),
               torch.randn(B, 32, 16, 16, requires_grad=True),
               torch.randn(B, 64,  8,  8, requires_grad=True)]
    t_feats = [torch.randn(B, 16, 32, 32),
               torch.randn(B, 32, 16, 16),
               torch.randn(B, 64,  8,  8)]

    loss_fn = DistillationLoss(
        temperature=4.0, alpha=0.5, beta=100.0,
        teacher_channels=[16, 32, 64], student_channels=[16, 32, 64],
    )
    total, parts = loss_fn(s_logits, t_logits, labels, s_feats, t_feats)
    total.backward()

    print("Loss smoke test passed")
    print(f"  total: {parts['total']:.4f}")
    print(f"  ce:    {parts['ce']:.4f}")
    print(f"  kd:    {parts['kd']:.4f}")
    print(f"  feat:  {parts['feat']:.6f}")
    print(f"  grad on student logits: {s_logits.grad.abs().mean().item():.6f}")
