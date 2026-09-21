"""CE gradient fitting with LoRA adapters."""

import json
from pathlib import Path

import torch

from proposals.ce_mle import CEMLEProposal, sequence_loglik
from proposals.iu import IUProposal
from model_loading import load_trainable_components


class CEMLELoRAProposal(IUProposal):
    MODE = "lora"
    likelihood_fitting = True

    configure_fit = CEMLEProposal.configure_fit
    configure_scoring = CEMLEProposal.configure_scoring
    configure_stopping = CEMLEProposal.configure_stopping
    check_event_rate_stop = CEMLEProposal.check_event_rate_stop
    update = CEMLEProposal.update

    def __init__(self, *, model, tokenizer, device, model_source,
                 elite_ratio=0.15, score_batch_size=16, fit_learning_rate=0.001,
                 fit_steps=1, stop_event_rate=0.1, max_rounds=300,
                 importance_weighted=True, score_mode="surrogate", use_chat_template=False):
        super().__init__(model=model, tokenizer=tokenizer, device=device,
                         model_source=model_source, uses_lora=True,
                         use_chat_template=use_chat_template)
        if not 0 < elite_ratio <= 1 or score_batch_size < 1:
            raise ValueError("Invalid elite ratio or fitting microbatch size")
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("CE LoRA requires only adapter parameters to be trainable")
        self.model.to(device).eval()
        self.elite_ratio = elite_ratio
        self.importance_weighted = bool(importance_weighted)
        self.configure_scoring(score_batch_size, score_mode)
        self.configure_fit("gradient", fit_learning_rate, fit_steps)
        self.configure_stopping(stop_event_rate, max_rounds)
        self._frozen = False
        self.last_fit_metrics = {}
        self.adaptation_outcome = {}
        self.total_fit_likelihood_evaluations = 0

    @classmethod
    def add_argparse_group(cls, parser):
        CEMLEProposal.add_argparse_group(parser)

    @classmethod
    def from_args(cls, args):
        if not args.use_lora:
            raise ValueError("CE_MLE_LORA requires --use_lora")
        if args.load_proposal_from:
            raise ValueError("CE_MLE_LORA checkpoint loading is not supported by train.py")
        model, tokenizer, device, source, _ = load_trainable_components(args)
        weighting = args.ce_importance_weighted
        return cls(
            model=model, tokenizer=tokenizer, device=device, model_source=source,
            elite_ratio=args.ce_elite_ratio, score_batch_size=args.ce_mle_score_batch_size,
            fit_learning_rate=args.ce_mle_fit_lr, fit_steps=args.ce_mle_fit_steps,
            stop_event_rate=args.ce_mle_stop_event_rate, max_rounds=args.ce_mle_max_rounds,
            importance_weighted=True if weighting is None else weighting,
            score_mode=args.ce_mle_score, use_chat_template=args.use_chat_template,
        )

    def build_optimizer(self, lr):
        return None

    def _gradient_fit(self, ids, prefix_len, weights, base_logits):
        del base_logits
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        initial = [p.detach().clone() for p in parameters]
        incumbent = None
        self.model.eval()
        for _ in range(self.fit_steps):
            self.model.zero_grad(set_to_none=True)
            loss_value = 0.0
            with torch.enable_grad():
                for start in range(0, len(ids), self.score_batch_size):
                    end = start + self.score_batch_size
                    logits = self.compute_logits(ids[start:end])
                    loss = -(sequence_loglik(logits, ids[start:end], prefix_len)
                             @ weights[start:end])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite CE LoRA likelihood")
                    loss.backward()
                    loss_value += float(loss.detach())
                    del logits, loss
            if incumbent is None:
                incumbent = -loss_value
            grads = [p.grad for p in parameters if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise RuntimeError("Missing or non-finite CE LoRA gradients")
            gradient_norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt())
            with torch.no_grad():
                for p in parameters:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-self.fit_learning_rate)
                    if not torch.isfinite(p).all():
                        raise RuntimeError("Non-finite CE LoRA parameters")
        with torch.no_grad():
            selected = 0.0
            for start in range(0, len(ids), self.score_batch_size):
                end = start + self.score_batch_size
                selected += float(sequence_loglik(self.compute_logits(ids[start:end]),
                                                  ids[start:end], prefix_len) @ weights[start:end])
            step_norm = float(torch.stack([(p - old).square().sum()
                                          for p, old in zip(parameters, initial)]).sum().sqrt())
        if not torch.isfinite(torch.tensor(selected)):
            raise RuntimeError("Non-finite CE LoRA post-step likelihood")
        return dict(incumbent_objective=incumbent, selected_objective=selected,
                    objective_improvement=selected-incumbent,
                    gradient_steps=self.fit_steps, gradient_norm=gradient_norm,
                    fit_learning_rate=self.fit_learning_rate, parameter_step_norm=step_norm,
                    update_applied=True,
                    fit_likelihood_evaluations=(self.fit_steps+1)*len(ids),
                    gradient_backward_trajectories=self.fit_steps*len(ids))

    def freeze_for_eval(self):
        self._frozen = True
        super().freeze_for_eval()

    def save(self, output_dir):
        artifact = super().save(output_dir)
        metadata = dict(algorithm="ce_mle", parameterization="lora", score_mode=self.score_mode,
                        importance_weighted=self.importance_weighted,
                        elite_ratio=self.elite_ratio, score_batch_size=self.score_batch_size,
                        fit_method=self.fit_method, fit_learning_rate=self.fit_learning_rate,
                        fit_steps=self.fit_steps, update_steps=self.update_steps,
                        stop_event_rate=self.stop_event_rate, max_rounds=self.max_rounds,
                        total_fit_likelihood_evaluations=self.total_fit_likelihood_evaluations,
                        adaptation_outcome=self.adaptation_outcome)
        (Path(artifact)/"ce_mle.json").write_text(json.dumps(metadata, indent=2)+"\n")
        (Path(output_dir)/"ce_mle_adaptation.json").write_text(
            json.dumps(self.adaptation_outcome, indent=2)+"\n")
        return artifact
