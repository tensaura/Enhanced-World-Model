from typing import Any, cast
import torch
import torch.nn.functional as F

from memory import MemoryModel


class TemporalTransformer(MemoryModel):
    tags: frozenset = frozenset()

    def __init__(
        self,
        latent_dim: int = 4,
        action_dim: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 32,
        cpc_temperature: float = 0.07,
        **_kwargs: Any,
    ) -> None:
        super().__init__()

        assert d_model % nhead == 0

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.input_dim = latent_dim + action_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_len = max_len
        self.cpc_temperature = cpc_temperature

        self.memory_input_proj = torch.nn.Linear(self.input_dim, d_model)

        self.prior_proj = torch.nn.Linear(latent_dim + action_dim, d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = torch.nn.Linear(d_model, latent_dim)

        # CPC head: projects h_t to a prediction of z_{t+1} for InfoNCE training.
        # Kept as a separate projection from output_proj so the MSE and CPC objectives
        # can learn independent projections.
        self.cpc_head = torch.nn.Linear(d_model, latent_dim)

        self.seq_buffer = None
        self.seq_lengths = None

    def update_memory(self, z_t: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """
        z_t:   (B, latent_dim)
        a_prev:(B, action_dim)
        """

        B = z_t.size(0)
        device = z_t.device

        if self.seq_buffer is None:  # Second part is for type checker.
            self.seq_buffer = torch.zeros(B, self.max_len, self.d_model, device=device)
            self.seq_lengths = torch.zeros(B, dtype=torch.long, device=device)
        assert self.seq_buffer is not None and self.seq_lengths is not None

        x = torch.cat([z_t, a_prev], dim=-1)
        x = self.memory_input_proj(x).unsqueeze(1)

        self.seq_buffer = torch.roll(self.seq_buffer, shifts=-1, dims=1)
        self.seq_buffer[:, -1] = x.squeeze(1).detach()  # Store detached for history
        self.seq_lengths = torch.clamp(self.seq_lengths + 1, max=self.max_len)

        memory_in = self.seq_buffer.clone()  # Clone buffer (already detached entries)
        memory_in[:, -1] = x.squeeze(1)  # Replace last with grad-connected tensor

        mask = torch.arange(self.max_len, device=device).unsqueeze(0) >= self.seq_lengths.unsqueeze(
            1
        )

        memory_out = self.transformer(memory_in, src_key_padding_mask=mask)

        # Extract from position -1 where the new grad-connected data was placed
        h_t = memory_out[:, -1]

        return h_t

    def predict_next(self, z_t: torch.Tensor, a_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        Predicts z_{t+1} from (z_t, a_t, h_t)
        """

        x = torch.cat([z_t, a_t], dim=-1)  # (B, latent+action)
        x = self.prior_proj(x)  # (B, d_model)

        x = x + h_t

        z_next = self.output_proj(x)  # (B, latent_dim)
        return z_next

    def forward(
        self, z_t: torch.Tensor, a_prev: torch.Tensor, a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full step:
          h_t = update_memory(z_t, a_prev)
          z_next_pred = predict_next(z_t, a_t, h_t)
        """

        h_t = self.update_memory(z_t, a_prev)
        z_next_pred = self.predict_next(z_t, a_t, h_t)
        return z_next_pred, h_t

    def reset_env_memory(self, env_idx: int | torch.Tensor) -> None:
        env_idx = int(env_idx)
        assert self.seq_buffer is not None and self.seq_lengths is not None
        self.seq_buffer[env_idx].zero_()
        self.seq_lengths[env_idx] = 0

    def compute_cpc_loss(self, h_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE (CPC) loss: the transformer's hidden state h_t should predict z_{t+1}.

        Uses a symmetric InfoNCE objective — the correct (h_t, z_{t+1}) pair is the
        positive sample; all other pairs within the batch are negatives.

        Requires batch_size >= 2; returns zero for smaller batches.

        Args:
            h_t:    Hidden state from update_memory (B, d_model).
            z_next: Actual next latent from the vision encoder (B, latent_dim).

        Returns:
            Scalar InfoNCE loss.
        """
        B = h_t.size(0)
        if B < 2:
            return torch.tensor(0.0, device=h_t.device)

        predictions = self.cpc_head(h_t)  # (B, latent_dim)

        # L2-normalise both sides for cosine similarity
        pred = F.normalize(predictions, dim=-1)
        target = F.normalize(z_next, dim=-1)

        # (B, B) similarity matrix; diagonal = positive pair
        logits = pred @ target.T / self.cpc_temperature

        labels = torch.arange(B, device=h_t.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss

    def export_hyperparams(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "max_len": self.max_len,
            "cpc_temperature": self.cpc_temperature,
        }

    def save_state(self) -> dict[str, torch.Tensor]:
        return cast(dict[str, Any], self.state_dict())

    def load(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state_dict)
