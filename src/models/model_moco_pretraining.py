import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_feat_alignment import ResNet18Encoder


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head used by MoCo.
    """

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MoCoPretrainingModel(nn.Module):
    """
    MoCo wrapper around the existing ResNet18 encoder.
    """

    def __init__(
        self,
        projection_dim=128,
        projector_hidden_dim=512,
        queue_size=16384,
        momentum=0.999,
        temperature=0.07,
    ):
        super().__init__()
        self.queue_size = int(queue_size)
        self.momentum = float(momentum)
        self.temperature = float(temperature)

        # Start from ImageNet weights, then adapt the encoder to mammograms
        # with MoCo before downstream risk fine-tuning.
        self.encoder_q = ResNet18Encoder()
        self.encoder_k = copy.deepcopy(self.encoder_q)

        self.projector_q = ProjectionHead(
            in_dim=512,
            hidden_dim=projector_hidden_dim,
            out_dim=projection_dim,
        )
        self.projector_k = copy.deepcopy(self.projector_q)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        queue = torch.randn(projection_dim, self.queue_size)
        queue = F.normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _expand_channels(img):
        return img if img.shape[1] == 3 else img.repeat(1, 3, 1, 1)

    def _encode(self, images, encoder, projector):
        features = encoder(self._expand_channels(images))
        pooled = self.global_pool(features).flatten(1)
        projected = projector(pooled)
        return F.normalize(projected, dim=1)

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())

        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size :].T)
            self.queue_ptr[0] = 0
            return

        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[:, ptr:end_ptr] = keys.T
        else:
            first_chunk = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_chunk].T
            self.queue[:, : end_ptr - self.queue_size] = keys[first_chunk:].T

        self.queue_ptr[0] = end_ptr % self.queue_size

    def compute_loss(self, view_q, view_k):
        q = self._encode(view_q, self.encoder_q, self.projector_q)

        with torch.no_grad():
            self.momentum_update_key_encoder()
            k = self._encode(view_k, self.encoder_k, self.projector_k)

        positive_logits = torch.einsum("nc,nc->n", q, k).unsqueeze(1)
        negative_logits = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        positive_similarity = positive_logits.mean().detach()
        negative_similarity = negative_logits.mean().detach()

        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
        }

    def forward(self, view_q, view_k):
        return self.compute_loss(view_q, view_k)

    def get_encoder_state_dict(self):
        return self.encoder_q.state_dict()
