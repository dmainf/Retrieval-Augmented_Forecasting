"""Cross-RAG retrieval-fusion head on top of the installed Chronos-Bolt model.

We subclass ``chronos.chronos_bolt.ChronosBoltModelForForecasting`` (which is
compatible with current ``transformers``) and reuse its ``encode`` / ``decode``.
On top of the decoded query representation we add the Cross-RAG fusion of
Lee et al. (2026):

  * cross branch  : query  <- retrieved (cross-attention, keys = retrieved x,
                    values = retrieved y)
  * self branch   : retrieved-y self-attention summary (mean-pooled)
  * mix           : ``mix_lambda * cross + (1 - mix_lambda) * self``

``seq_len`` and ``mix_lambda`` are read from the ``INPUT_LEN`` / ``LAMBDA`` env
vars at construction (set by the cross_raf method), mirroring the original code.
"""
import os

import torch
import torch.nn as nn

from chronos.chronos_bolt import ChronosBoltModelForForecasting, ChronosBoltOutput


class ChronosBoltModelForForecastingWithRetrieval(ChronosBoltModelForForecasting):
    def __init__(self, config, augment: str = "moe"):
        super().__init__(config)
        self.augment = augment
        d = config.d_model
        self.mix_lambda = float(os.getenv("LAMBDA", "0.7"))
        self.context_length_in = int(os.getenv("INPUT_LEN", "512"))
        pred = self.chronos_config.prediction_length

        self.encode_mlp_x = nn.Sequential(nn.Linear(self.context_length_in, d), nn.ReLU(), nn.Linear(d, d))
        self.encode_mlp_y = nn.Sequential(nn.Linear(pred, d), nn.ReLU(), nn.Linear(d, d))
        self.cross_mha = nn.MultiheadAttention(embed_dim=d, num_heads=8, batch_first=True)
        self.self_mha = nn.MultiheadAttention(embed_dim=d, num_heads=8, batch_first=True)
        self.ffn_cross = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.ffn_self = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.dropout = nn.Dropout(p=0.2)

    def init_extra_weights(self, layers):
        factor = self.config.initializer_factor
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                self.init_extra_weights(layer)
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=factor * ((self.chronos_config.input_patch_size * 2) ** -0.5))
                if layer.bias is not None:
                    layer.bias.data.zero_()
            if isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.xavier_uniform_(layer.out_proj.weight)
                if layer.in_proj_bias is not None:
                    layer.in_proj_bias.data.zero_()
                if layer.out_proj.bias is not None:
                    layer.out_proj.bias.data.zero_()

    def forward(
        self,
        context: torch.Tensor,
        mask=None,
        target=None,
        target_mask=None,
        retrieved_seq=None,
        distances=None,
    ) -> ChronosBoltOutput:
        batch_size = context.size(0)
        hidden_states, loc_scale, input_embeds, attention_mask = self.encode(context=context, mask=mask)
        sequence_output = self.decode(input_embeds, attention_mask, hidden_states)  # (B, 1, d)

        # normalize retrieved windows (B, K, win_len) and split into x / y
        retrieved_seq, _ = self.instance_norm(retrieved_seq)
        L = self.chronos_config.prediction_length
        r_L = retrieved_seq.shape[-1]
        assert (r_L - L) == self.context_length_in, (
            f"retrieved context length {r_L - L} != seq_len {self.context_length_in}"
        )
        retrieved_x, retrieved_y = retrieved_seq.split((r_L - L, L), dim=2)
        retrieved_x = retrieved_x.to(sequence_output.dtype)
        retrieved_y = retrieved_y.to(sequence_output.dtype)

        if self.augment == "moe":
            B, K, Lx = retrieved_x.shape
            Ly = retrieved_y.shape[2]
            # encode all K retrieved windows in one batched pass (B*K) -> (B, K, d)
            rx = self.encode_mlp_x(retrieved_x.reshape(B * K, Lx)).reshape(B, K, -1)
            ry = self.encode_mlp_y(retrieved_y.reshape(B * K, Ly)).reshape(B, K, -1)

            # cross branch: query attends to retrieved (keys=rx, values=ry)
            cross_out, _ = self.cross_mha(sequence_output, rx, ry)
            cross_out = sequence_output + cross_out
            cross_out = cross_out + self.dropout(self.ffn_cross(cross_out))

            # self branch: retrieved-y self-attention then mean pool
            self_out, _ = self.self_mha(ry, ry, ry)
            self_out = ry + self_out
            self_out = self_out + self.dropout(self.ffn_self(self_out))
            self_pool = self_out.mean(dim=1, keepdim=True)  # B,1,d

            sequence_output = self.mix_lambda * cross_out + (1 - self.mix_lambda) * self_pool
        else:
            raise ValueError(f"Unsupported augment mode for retrieval head: {self.augment}")

        quantile_preds_shape = (batch_size, self.num_quantiles, self.chronos_config.prediction_length)
        quantile_preds = self.output_patch_embedding(sequence_output).view(*quantile_preds_shape)

        loss = None
        if target is not None:
            target, _ = self.instance_norm(target, loc_scale)
            target = target.unsqueeze(1)
            assert self.chronos_config.prediction_length >= target.shape[-1]
            target = target.to(quantile_preds.device)
            target_mask = (
                target_mask.unsqueeze(1).to(quantile_preds.device)
                if target_mask is not None else ~torch.isnan(target)
            )
            target[~target_mask] = 0.0
            if self.chronos_config.prediction_length > target.shape[-1]:
                padding_shape = (*target.shape[:-1], self.chronos_config.prediction_length - target.shape[-1])
                target = torch.cat([target, torch.zeros(padding_shape).to(target)], dim=-1)
                target_mask = torch.cat([target_mask, torch.zeros(padding_shape).to(target_mask)], dim=-1)
            loss = (
                2 * torch.abs(
                    (target - quantile_preds)
                    * ((target <= quantile_preds).float() - self.quantiles.view(1, self.num_quantiles, 1))
                ) * target_mask.float()
            )
            loss = loss.mean(dim=-2).sum(dim=-1).mean()

        quantile_preds = self.instance_norm.inverse(
            quantile_preds.view(batch_size, -1), loc_scale
        ).view(*quantile_preds_shape)

        return ChronosBoltOutput(loss=loss, quantile_preds=quantile_preds)
