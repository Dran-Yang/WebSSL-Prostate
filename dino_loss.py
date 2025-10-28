"""
DINO self-supervised loss with temperature scheduling and output centering.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float,
        teacher_temp: float,
        warmup_teacher_epochs: int,
        total_epochs: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, out_dim))

        warmup_teacher_temp = float(warmup_teacher_temp)
        teacher_temp = float(teacher_temp)
        warmup_teacher_epochs = max(0, int(warmup_teacher_epochs))

        if warmup_teacher_epochs > 0:
            warmup_schedule = torch.linspace(
                warmup_teacher_temp,
                teacher_temp,
                warmup_teacher_epochs,
            )
            rest = torch.ones(total_epochs - warmup_teacher_epochs) * teacher_temp
            self.teacher_temp_schedule = torch.cat([warmup_schedule, rest])
        else:
            self.teacher_temp_schedule = torch.ones(total_epochs) * teacher_temp

    def forward(
        self,
        student_outputs: Iterable[torch.Tensor],
        teacher_outputs: Iterable[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        student_out = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs]

        epoch = min(int(epoch), len(self.teacher_temp_schedule) - 1)
        temp = float(self.teacher_temp_schedule[epoch])

        teacher_out: List[torch.Tensor] = []
        for t in teacher_outputs:
            logits = (t - self.center) / temp
            teacher_out.append(F.softmax(logits, dim=-1).detach())

        total_loss = 0.0
        num_terms = 0
        for iq, q in enumerate(teacher_out):
            for v, s in enumerate(student_out):
                if v == iq:
                    continue
                total_loss += torch.sum(-q * s, dim=-1).mean()
                num_terms += 1

        self.update_center(teacher_outputs)

        if num_terms == 0:
            raise ValueError("No loss terms computed. Ensure global crops are included.")
        return total_loss / num_terms

    @torch.no_grad()
    def update_center(self, teacher_outputs: Iterable[torch.Tensor]) -> None:
        concat = torch.cat(list(teacher_outputs), dim=0)
        mean = torch.mean(concat, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + mean * (1.0 - self.center_momentum)
