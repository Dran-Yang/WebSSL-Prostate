"""
Model builder for Web-DINO style student/teacher networks.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

# from webssl.dinov2.vision_transformer import build_vit

try:
    from transformers import Dinov2Model
except ImportError:  # pragma: no cover - optional dependency
    Dinov2Model = None


class Dinov2Backbone(nn.Module):
    """Thin wrapper around Hugging Face Dinov2Model returning CLS embeddings."""

    def __init__(self, model: "Dinov2Model") -> None:  # type: ignore[name-defined]
        super().__init__()
        self.model = model
        self.embed_dim: int = int(getattr(model.config, "hidden_size", 0))
        self.patch_size: Optional[int] = getattr(model.config, "patch_size", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = x if torch.is_floating_point(x) else x.float()
        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        return hidden[:, 0]


class DINOProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        if norm_last_layer:
            self.last_layer.weight_g.data.fill_(1.0)
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class StudentTeacher(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_head: nn.Module,
        teacher_head: nn.Module,
    ) -> None:
        super().__init__()
        self.student_backbone = student
        self.teacher_backbone = teacher
        self.student_head = student_head
        self.teacher_head = teacher_head

        self.teacher_backbone.requires_grad_(False)
        self.teacher_head.requires_grad_(False)
        self.teacher_backbone.eval()
        self.teacher_head.eval()

    def forward_student(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.student_backbone(images)
        return self.student_head(feats)

    @torch.no_grad()
    def forward_teacher(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.teacher_backbone(images)
        return self.teacher_head(feats)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student_param, teacher_param in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            teacher_param.data = teacher_param.data * momentum + student_param.data * (1.0 - momentum)
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            teacher_param.data = teacher_param.data * momentum + student_param.data * (1.0 - momentum)


def build_webdino(
    cfg: Dict[str, object],
    image_size: int,
) -> StudentTeacher:
    student_cfg = dict(cfg.get("student") or {})
    teacher_cfg = dict(cfg.get("teacher") or {})
    head_cfg = dict(cfg.get("head") or {})

    hf_model_name = cfg.get("hf_model_name")
    hf_cache_dir = cfg.get("hf_cache_dir")

    if hf_model_name:
        if Dinov2Model is None:
            raise ImportError(
                "transformers is required to load Hugging Face weights. "
                "Install it with `pip install transformers`."
            )
        student_core = Dinov2Model.from_pretrained(
            hf_model_name,
            cache_dir=hf_cache_dir,
        )
        teacher_core = Dinov2Model.from_pretrained(
            hf_model_name,
            cache_dir=hf_cache_dir,
        )
        teacher_core.load_state_dict(student_core.state_dict())
        student = Dinov2Backbone(student_core)
        teacher = Dinov2Backbone(teacher_core)
        embed_dim = student.embed_dim
    # else:
    #     base_kwargs = {
    #         "image_size": image_size,
    #         "patch_size": int(cfg.get("patch_size", 16)),
    #         "in_channels": int(cfg.get("in_channels", 3)),
    #         "embed_dim": int(cfg.get("embed_dim", 384)),
    #         "depth": int(cfg.get("depth", 12)),
    #         "num_heads": int(cfg.get("num_heads", 6)),
    #         "mlp_ratio": float(cfg.get("mlp_ratio", 4.0)),
    #         "qkv_bias": bool(cfg.get("qkv_bias", True)),
    #         "drop_path_rate": float(student_cfg.get("drop_path_rate", 0.1)),
    #         "global_pool": "token",
    #     }
    #     student = build_vit(**base_kwargs)
    #     teacher = build_vit(**{**base_kwargs, "drop_path_rate": 0.0})
    #     teacher.load_state_dict(student.state_dict())
    #     embed_dim = base_kwargs["embed_dim"]

    student_head = DINOProjectionHead(
        in_dim=embed_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
        out_dim=int(cfg.get("out_dim", 65536)),
        norm_last_layer=bool(head_cfg.get("norm_last_layer", True)),
    )
    teacher_head = DINOProjectionHead(
        in_dim=embed_dim,
        hidden_dim=int(head_cfg.get("hidden_dim", 2048)),
        bottleneck_dim=int(head_cfg.get("bottleneck_dim", 256)),
        out_dim=int(cfg.get("out_dim", 65536)),
        norm_last_layer=False,
    )
    teacher_head.load_state_dict(student_head.state_dict())

    model = StudentTeacher(
        student=student,
        teacher=teacher,
        student_head=student_head,
        teacher_head=teacher_head,
    )

    momentum_schedule = {
        "base": float(teacher_cfg.get("momentum_base", 0.996)),
        "warmup_epochs": int(teacher_cfg.get("momentum_warmup_epochs", 10)),
    }
    model.momentum_schedule = momentum_schedule  # type: ignore[attr-defined]
    model.embed_dim = embed_dim  # type: ignore[attr-defined]

    return model
