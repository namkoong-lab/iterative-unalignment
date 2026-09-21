"""Frozen language-model backend for TPS."""

import numpy as np
import torch


class LanguageModelBackend:
    def __init__(self, model, prefix_ids, event, length, score_mode="auto", use_cache=False):
        if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1 or prefix_ids.shape[1] == 0:
            raise ValueError("TPS requires one nonempty, unpadded prompt")
        if length <= 0:
            raise ValueError("length must be positive")
        if score_mode not in {"auto", "event_score", "base_surrogate"}:
            raise ValueError("Unknown score_mode")
        if score_mode == "auto":
            score_mode = "event_score" if callable(getattr(event, "compute_score", None)) else "base_surrogate"
        if score_mode == "event_score" and not callable(getattr(event, "compute_score", None)):
            raise ValueError("This event has no compute_score; use base_surrogate")
        self.model = model.eval()
        self.prefix_ids = prefix_ids
        self.device = prefix_ids.device
        self.event = event
        self.length = int(length)
        self.score_mode = score_mode
        self.use_cache = bool(use_cache)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def sample(self, batch_size):
        empty = np.empty((batch_size, self.length), dtype=np.int64)
        return self.regenerate(empty, 0)

    @torch.inference_mode()
    def regenerate(self, states, cut):
        if not 0 <= cut < self.length:
            raise ValueError("cut must include at least one generated token")
        if states.ndim != 2 or states.shape[1] != self.length:
            raise ValueError("states must have shape [batch, completion_length]")
        kept = torch.as_tensor(states[:, :cut], dtype=torch.long, device=self.device)
        ids = torch.cat((self.prefix_ids.expand(len(states), -1), kept), dim=1)
        cache = None
        for _ in range(self.length - cut):
            inputs = ids[:, -1:] if cache is not None else ids
            kwargs = {"input_ids": inputs, "use_cache": self.use_cache}
            if cache is not None:
                kwargs["past_key_values"] = cache
            output = self.model(**kwargs)
            token = torch.distributions.Categorical(logits=output.logits[:, -1, :]).sample()
            ids = torch.cat((ids, token[:, None]), dim=1)
            if self.use_cache:
                cache = output.past_key_values
        return ids[:, self.prefix_ids.shape[1]:].cpu().numpy()

    @torch.inference_mode()
    def evaluate(self, states):
        generated = torch.as_tensor(states, dtype=torch.long, device=self.device)
        indicator = self.event.compute_indicator(generated)
        if self.score_mode == "event_score":
            score = self.event.compute_score(generated)
        else:
            ids = torch.cat((self.prefix_ids.expand(len(states), -1), generated), dim=1)
            logits = self.model(input_ids=ids, use_cache=False).logits
            score = -self.event.compute_surrogate_loss(
                logits, self.prefix_ids.shape[1], generated_tokens=generated,
            )
        return score.double().cpu().numpy(), indicator.double().cpu().numpy()
