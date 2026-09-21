"""CE gradient fitting of elite continuation likelihoods."""

import json
import math
import os

import torch

from is_weights import generated_predictor_logits
from proposals.ce_steering import CEProposal
from proposals._ce_common import elite_weights


def elite_indices(scores, elite_ratio, threshold=None):
    """Higher scores are better. Event thresholds include all boundary ties."""
    if scores.ndim != 1 or scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("CE fitting requires a nonempty vector of finite scores")
    if threshold is None:
        count = max(1, round(scores.numel() * elite_ratio))
        indices = torch.argsort(scores, descending=True, stable=True)[:count]
        return indices, float(scores[indices[-1]])
    if not math.isfinite(threshold):
        raise ValueError("Event threshold must be finite")
    cutoff = min(threshold, float(torch.quantile(scores.double(), 1 - elite_ratio)))
    return torch.nonzero(scores >= cutoff, as_tuple=True)[0], cutoff


def sequence_loglik(logits, sampled_ids, prefix_len):
    """Sum log probabilities of the continuation, conditional on the prompt."""
    predictors = generated_predictor_logits(logits, prefix_len).float()
    targets = sampled_ids[:, prefix_len:]
    token_logp = predictors.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp - torch.logsumexp(predictors, dim=-1)
    return token_logp.double().sum(dim=-1)


class CEMLEProposal(CEProposal):
    likelihood_fitting = True

    def __init__(self, *, score_batch_size=8, importance_weighted=True,
                 score_mode="surrogate", stop_event_rate=0.1, max_rounds=300,
                 fit_method="gradient", fit_learning_rate=0.01, fit_steps=1, **kwargs):
        super().__init__(importance_weighted=importance_weighted, **kwargs)
        if self.steering_init_scale != 0 or self.generation_temperature != 1:
            raise ValueError("CE_MLE requires zero initialization and generation temperature 1")
        if not self.eval_use_mean_only:
            raise ValueError("CE_MLE always generates from the current vector, including eval")
        self.configure_scoring(score_batch_size, score_mode)
        self.configure_fit(fit_method, fit_learning_rate, fit_steps)
        self.last_fit_metrics = {}
        self.total_fit_likelihood_evaluations = 0
        self.configure_stopping(stop_event_rate, max_rounds)
        self.adaptation_outcome = {}

    def configure_stopping(self, stop_event_rate, max_rounds):
        if not math.isfinite(stop_event_rate) or not 0 <= stop_event_rate <= 1:
            raise ValueError("CE_MLE event-rate target must be in [0, 1] (0 disables it)")
        if max_rounds < 1:
            raise ValueError("CE_MLE max rounds must be positive")
        self.stop_event_rate = float(stop_event_rate)
        self.max_rounds = int(max_rounds)

    def check_event_rate_stop(self, *, step, rare_event_count, batch_size):
        """Check the event-rate target before updating the proposal."""
        if batch_size < 1 or not 0 <= rare_event_count <= batch_size:
            raise ValueError("Invalid event count or batch size")
        rate = rare_event_count / batch_size
        reached = self.stop_event_rate > 0 and rate >= self.stop_event_rate
        self.adaptation_outcome.update(
            target_event_rate=self.stop_event_rate, last_observed_round=step,
            last_training_event_rate=rate, last_training_event_count=rare_event_count,
            batch_size=batch_size, target_reached=reached,
            best_training_event_rate=max(rate, self.adaptation_outcome.get("best_training_event_rate", 0)),
            status="event_rate_target_reached" if reached else "adapting",
        )
        if reached:
            self.last_fit_metrics = {
                "update_skipped": True,
                "objective_improvement": 0.0, "gradient_steps": 0,
            }
            return True, (f"training event rate {rare_event_count}/{batch_size} = {rate:.2%} "
                          f"reached target {self.stop_event_rate:.2%}; keeping the generating vector")
        return False, None

    def should_stop(self, **kwargs):
        return False, None

    def configure_scoring(self, score_batch_size, score_mode):
        if score_batch_size < 1:
            raise ValueError("CE_MLE requires score batch size >= 1")
        if score_mode not in ("surrogate", "event"):
            raise ValueError("CE_MLE score must be surrogate or event")
        self.score_batch_size = int(score_batch_size)
        self.score_mode = score_mode

    def configure_fit(self, method, learning_rate, steps):
        if method != "gradient":
            raise ValueError("CE_MLE uses gradient fitting")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("CE_MLE fit learning rate must be positive and finite")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("CE_MLE fit steps must be a positive integer")
        self.fit_method = method
        self.fit_learning_rate = float(learning_rate)
        self.fit_steps = steps

    @classmethod
    def add_argparse_group(cls, parser):
        super().add_argparse_group(parser)
        if getattr(parser, "_ce_mle_args_registered", False):
            return
        parser._ce_mle_args_registered = True
        group = parser.add_argument_group("CE_MLE likelihood fitting")
        group.add_argument("--ce_mle_score_batch_size", type=int, default=8,
                           help="Maximum elite trajectories per fitting microbatch.")
        group.add_argument("--ce_mle_score", choices=("surrogate", "event"), default="surrogate",
                           help="Frozen proposal surrogate ranking, or literal event threshold CE.")
        group.add_argument("--ce_mle_stop_event_rate", type=float, default=0.1,
                           help="Stop before fitting when the batch event fraction reaches this target; 0 disables event-rate stopping.")
        group.add_argument("--ce_mle_max_rounds", type=int, default=300,
                           help="Hard adaptation-round cap, also bounded by --steps.")
        group.add_argument("--ce_mle_fit_method", choices=("gradient",), default="gradient")
        group.add_argument("--ce_mle_fit_lr", type=float, default=0.01,
                           help="SGD step size on elite-weighted, sequence-summed negative log likelihood.")
        group.add_argument("--ce_mle_fit_steps", type=int, default=1,
                           help="Global SGD steps on the frozen elite batch per CE round.")

    @classmethod
    def from_args(cls, args):
        proposal = super().from_args(args)
        proposal.configure_scoring(args.ce_mle_score_batch_size, args.ce_mle_score)
        proposal.configure_stopping(getattr(args, "ce_mle_stop_event_rate", 0.1),
                                    getattr(args, "ce_mle_max_rounds", 300))
        proposal.configure_fit(getattr(args, "ce_mle_fit_method", "gradient"),
                               getattr(args, "ce_mle_fit_lr", 0.01),
                               getattr(args, "ce_mle_fit_steps", 1))
        artifact = getattr(args, "load_proposal_from", None)
        if artifact:
            proposal.load(artifact)
        return proposal

    def _sample_candidates(self, batch_size):
        return self.mu.unsqueeze(0).expand(batch_size, -1).contiguous()

    @torch.no_grad()
    def _likelihood_objective(self, vector, ids, prefix_len, weights, base_logits=None):
        previous = self._wrapper.current_vectors
        objective = torch.zeros((), device=self.device, dtype=torch.float64)
        try:
            for start in range(0, len(ids), self.score_batch_size):
                end = start + self.score_batch_size
                batch_ids = ids[start:end]
                if self.MODE == "logit" and base_logits is not None:
                    logits = base_logits[start:end] + vector[None, None, :]
                else:
                    self._wrapper.set_current_vectors(vector.expand(len(batch_ids), -1))
                    logits = self._wrapper(input_ids=batch_ids).logits
                objective += sequence_loglik(logits, batch_ids, prefix_len) @ weights[start:end]
        finally:
            self._wrapper.set_current_vectors(previous)
        return objective

    def _gradient_fit(self, ids, prefix_len, weights, base_logits):
        """Accumulate microbatch gradients with globally normalized elite weights."""
        previous_vectors = self._wrapper.current_vectors
        initial = self.mu.detach().clone()
        theta = initial.clone().requires_grad_(True)
        incumbent = None
        try:
            for _ in range(self.fit_steps):
                theta.grad = None
                loss_value = 0.0
                with torch.enable_grad():
                    for start in range(0, len(ids), self.score_batch_size):
                        stop = min(start + self.score_batch_size, len(ids))
                        chunk = ids[start:stop]
                        if self.MODE == "logit" and base_logits is not None:
                            logits = base_logits[start:stop] + theta[None, None, :]
                        else:
                            self._wrapper.set_current_vectors(theta.expand(len(chunk), -1))
                            logits = self._wrapper(input_ids=chunk).logits
                        loss = -(sequence_loglik(logits, chunk, prefix_len) @ weights[start:stop])
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite CE_MLE gradient loss")
                        loss.backward()
                        loss_value += float(loss.detach())
                        del loss, logits
                if incumbent is None:
                    incumbent = -loss_value
                if theta.grad is None or not torch.isfinite(theta.grad).all():
                    raise RuntimeError("Non-finite or missing CE_MLE steering gradient")
                gradient_norm = float(theta.grad.norm())
                with torch.no_grad():
                    theta.add_(theta.grad, alpha=-self.fit_learning_rate)
                    if not torch.isfinite(theta).all():
                        raise RuntimeError("Non-finite CE_MLE gradient update")
            selected = float(self._likelihood_objective(
                theta.detach(), ids, prefix_len, weights, base_logits))
            if not math.isfinite(selected):
                raise RuntimeError("Non-finite CE_MLE post-step objective")
        finally:
            self._wrapper.set_current_vectors(previous_vectors)
        self.mu = theta.detach().clone()
        return dict(incumbent_objective=incumbent, selected_objective=selected,
                    objective_improvement=selected-incumbent,
                    gradient_steps=self.fit_steps, gradient_norm=gradient_norm,
                    fit_learning_rate=self.fit_learning_rate,
                    parameter_step_norm=float((self.mu-initial).norm()),
                    update_applied=True,
                    fit_likelihood_evaluations=(self.fit_steps+1)*len(ids),
                    gradient_backward_trajectories=self.fit_steps*len(ids))

    @torch.no_grad()
    def update(self, main_loss_per_sample, reg_loss, lambda_value,
               log_importance_weights=None, sampled_ids=None, prefix_len=None,
               score_threshold=None, original_logits=None, elite_logits_provider=None):
        del reg_loss, lambda_value
        if self._frozen:
            raise RuntimeError("Cannot fit a frozen CE_MLE proposal")
        if sampled_ids is None or prefix_len is None:
            raise ValueError("CE_MLE fitting requires trajectories and prefix length")
        scores = -main_loss_per_sample.detach().to(self.device)
        if scores.ndim != 1 or scores.shape[0] != sampled_ids.shape[0]:
            raise ValueError("CE_MLE scores, trajectories, and weights must share a batch")
        if self.score_mode == "event" and score_threshold is None:
            raise ValueError("Event-score CE requires a threshold")
        indices, cutoff = elite_indices(scores, self.elite_ratio, score_threshold)
        elite_ids = sampled_ids.detach()[indices].clone()
        weights = elite_weights(indices, log_importance_weights, weighted=self.importance_weighted)
        cached = None
        if self.MODE == "logit" and original_logits is not None:
            cached = original_logits.detach()[indices]
        elif self.MODE == "logit" and elite_logits_provider is not None:
            cached = elite_logits_provider(elite_ids).detach()
            if cached.shape[:2] != elite_ids.shape:
                raise ValueError("Elite reference logits do not match elite trajectories")
        fit = self._gradient_fit(elite_ids, prefix_len, weights, cached)
        self.update_steps += 1
        self.total_fit_likelihood_evaluations += fit['fit_likelihood_evaluations']
        preview_order = torch.argsort(scores[indices], descending=True, stable=True)[:3]
        elite_samples = [
            dict(rank=rank, batch_index=int(indices[j]), score=float(scores[indices[j]]),
                 weight=float(weights[j]), token_ids=elite_ids[j, prefix_len:].cpu().tolist())
            for rank, j in enumerate(preview_order.tolist(), start=1)
        ]
        self.last_fit_metrics = {
            **fit, "fit_method": self.fit_method, "importance_weighted": self.importance_weighted,
            "score_mode": self.score_mode,
            "elite_count": indices.numel(), "score_cutoff": cutoff,
            "elite_weight_ess": float(1 / weights.square().sum()),
            "total_fit_likelihood_evaluations": self.total_fit_likelihood_evaluations,
            "cached_logit_scoring": cached is not None,
            "elite_samples": elite_samples,
        }
        return main_loss_per_sample.detach().mean()

    def save(self, output_dir):
        artifact = super().save(output_dir)
        metadata = {
            "algorithm": "ce_mle", "steering_mode": self.MODE,
            "importance_weighted": self.importance_weighted, "score_batch_size": self.score_batch_size,
            "score_mode": self.score_mode,
            "fit_method": self.fit_method, "fit_learning_rate": self.fit_learning_rate,
            "fit_steps": self.fit_steps,
            "stop_event_rate": self.stop_event_rate, "max_rounds": self.max_rounds,
            "adaptation_outcome": self.adaptation_outcome,
            "total_fit_likelihood_evaluations": self.total_fit_likelihood_evaluations,
        }
        with open(os.path.join(artifact, "ce_mle.json"), "w") as handle:
            json.dump(metadata, handle, indent=2)
        with open(os.path.join(output_dir, "ce_mle_adaptation.json"), "w") as handle:
            json.dump(self.adaptation_outcome, handle, indent=2)
        return artifact

    def load(self, artifact):
        with open(os.path.join(artifact, "ce_mle.json")) as handle:
            metadata = json.load(handle)
        state = torch.load(os.path.join(artifact, "steering_vector.pt"),
                           map_location=self.device, weights_only=True)
        if metadata["steering_mode"] != self.MODE or state["param_dim"] != self.param_dim:
            raise ValueError("Saved CE_MLE intervention family does not match")
        if state["model_source"] != self.model_source:
            raise ValueError("Saved CE_MLE base model does not match")
        self.mu = state["steering_vector"].to(self.device).clone()
        self.sigma_init = state["sigma_init"]
        self.sigma.fill_(self.sigma_init)
        self.elite_ratio = state["elite_ratio"]
        self.update_steps = state["update_steps"]
        self.configure_scoring(metadata["score_batch_size"], metadata["score_mode"])
        self.importance_weighted = metadata.get("importance_weighted", True)
        self.configure_stopping(metadata["stop_event_rate"], metadata["max_rounds"])
        self.adaptation_outcome = metadata.get("adaptation_outcome", {})
        self.configure_fit(metadata.get("fit_method", "gradient"),
                           metadata.get("fit_learning_rate", 0.01), metadata.get("fit_steps", 1))
        self.total_fit_likelihood_evaluations = metadata["total_fit_likelihood_evaluations"]


class CEMLEActivationProposal(CEMLEProposal):
    MODE = "activation"


class CEMLELogitProposal(CEMLEProposal):
    MODE = "logit"
