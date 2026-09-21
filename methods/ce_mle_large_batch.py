"""CE gradient fitting with memory-bounded token batches."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import pickle
import random
import time

import numpy as np
import torch

from events.token_event import TokenEvent
from is_weights import generated_predictor_logits, ordinary_is_log_estimate
from model_loading import _load_causal_lm, _load_tokenizer, _setup_lora_model
from metrics_logging import MetricLogger
from proposals.ce_mle import CEMLEActivationProposal, CEMLELogitProposal, sequence_loglik
from proposals.ce_mle_lora import CEMLELoRAProposal


def now(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.no_grad()
def collect_batch(proposal, reference, event, prefix, batch_size, length, microbatch):
    """Generate every row from current mu; no fitting inside the chunk loop."""
    if min(batch_size, length, microbatch) < 1:
        raise ValueError('Batch sizes and length must be positive')
    parts = {key: [] for key in ('ids', 'loss', 'log_p', 'log_q', 'event', 'target_prob')}
    costs = dict(q_inference=0., q_forward=0., p_forward=0., diagnostics=0.)
    for start in range(0, batch_size, microbatch):
        t = now(proposal.device)
        ids, prefix_len = proposal.rollout_batch(prefix, min(microbatch, batch_size-start), length)
        t1 = now(proposal.device)
        q_logits = proposal.compute_logits(ids)
        t2 = now(proposal.device)
        p_logits = reference(input_ids=ids).logits
        t3 = now(proposal.device)
        parts['ids'].append(ids)
        parts['log_q'].append(sequence_loglik(q_logits, ids, prefix_len))
        parts['log_p'].append(sequence_loglik(p_logits, ids, prefix_len))
        parts['loss'].append(event.compute_surrogate_loss(q_logits, prefix_len, ids[:, prefix_len:]))
        parts['event'].append(event.compute_indicator(ids[:, prefix_len:]))
        predictors = generated_predictor_logits(q_logits, prefix_len).float()
        target_logp = predictors[:, :, event.token_id] - torch.logsumexp(predictors, dim=-1)
        parts['target_prob'].append(target_logp.exp().mean(-1))
        del q_logits, p_logits, predictors, target_logp
        t4 = now(proposal.device)
        for key, seconds in zip(costs, (t1-t, t2-t1, t3-t2, t4-t3)):
            costs[key] += seconds
    batch = {key: torch.cat(value) for key, value in parts.items()}
    batch['log_weights'] = batch['log_p'] - batch['log_q']
    batch['prefix_len'] = prefix_len
    batch['costs'] = costs
    return batch


@torch.no_grad()
def elite_reference_logits(reference, ids, microbatch):
    return torch.cat([reference(input_ids=chunk).logits for chunk in ids.split(microbatch)])


def metric_row(batch, step, fit_metrics, fit_seconds):
    log_est = float(ordinary_is_log_estimate(batch['log_weights'], batch['event']))
    logw = batch['log_weights'].double()
    normalized = torch.softmax(logw, 0)
    event_logw = logw[batch['event'].bool()]
    event_normalized = torch.softmax(event_logw, 0)
    ess = float(1/normalized.square().sum())
    event_ess = float(1/event_normalized.square().sum()) if len(event_logw) else None
    return dict(step=step, event_indicator=batch['event'].int().tolist(),
                log_importance_weights=batch['log_weights'].tolist(),
                log_p_seq=batch['log_p'].tolist(), log_q_seq=batch['log_q'].tolist(),
                main_loss=float(batch['loss'].mean()),
                rare_token_prob_mean=float(batch['target_prob'].mean()),
                is_estimate=float(torch.tensor(log_est, dtype=torch.float64).exp()), log_is_estimate=log_est,
                population_ess_count=ess, population_ess=ess/len(logw),
                rare_event_ess_count=event_ess,
                rare_event_ess=event_ess/len(event_logw) if len(event_logw) else None,
                max_normalized_weight=float(normalized.max()),
                max_event_contribution_fraction=float(event_normalized.max()) if len(event_logw) else None,
                log_weight_min=float(logw.min()), log_weight_max=float(logw.max()),
                unique_trajectory_fraction=len(torch.unique(batch['ids'], dim=0))/len(logw),
                sample_token_ids=batch['ids'][:3, batch['prefix_len']:].tolist(),
                event_sample_token_ids=batch['ids'][batch['event'].bool()][:3, batch['prefix_len']:].tolist(),
                proposal_metrics=dict(fit_metrics),
                timings={**batch['costs'], 'ce_fit': fit_seconds},
                event_rate=float(batch['event'].float().mean()))


def save_metrics(path, rows, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        pickle.dump(dict(per_step=rows, metadata=metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def run(config, output, proposal, reference, event, prefix):
    metadata = dict(config, ordinary_is=True, ordinary_is_unclipped=True, cost_includes_fitting=True,
                    runner='ce_mle_large_batch', dense_regularization_diagnostics=False,
                    metrics_burn_in=config.get('metrics_burn_in',20), metrics_window=50,
                    estimate_timing='generated before this round fitting',
                    fit_method=proposal.fit_method, fit_learning_rate=proposal.fit_learning_rate,
                    fit_steps=proposal.fit_steps, score_mode='surrogate')
    metadata['importance_weighted'] = proposal.importance_weighted
    def accuracy_logger(phase='train'):
        return MetricLogger(output_pickle_path=None, metadata=dict(metadata, phase=phase), log_every=1,
                            total_steps=config['max_rounds'] if phase == 'train' else config['eval_batches'],
                            tokenizer=proposal.tokenizer if hasattr(proposal.tokenizer, 'batch_decode') else None,
                            burn_in=metadata['metrics_burn_in'])
    accuracy = accuracy_logger()
    train_rows = []
    proposal.adaptation_outcome = dict(status='adapting', round_limit=config['max_rounds'])
    for step in range(1, config['max_rounds']+1):
        batch = collect_batch(proposal, reference, event, prefix, config['batch_size'],
                              config['length'], config['microbatch'])
        hits = int(batch['event'].sum())
        stop, reason = proposal.check_event_rate_stop(step=step, rare_event_count=hits,
                                                      batch_size=config['batch_size'])
        fit_start = now(proposal.device)
        if not stop:
            provider = None
            if proposal.MODE == 'logit':
                provider = lambda ids: elite_reference_logits(reference, ids, config['score_batch_size'])
            proposal.update(batch['loss'], None, 0., log_importance_weights=batch['log_weights'],
                            sampled_ids=batch['ids'], prefix_len=batch['prefix_len'],
                            elite_logits_provider=provider)
        fit_seconds = now(proposal.device)-fit_start
        proposal.last_fit_metrics.update(training_event_rate=hits/config['batch_size'],
                                         target_event_rate=proposal.stop_event_rate,
                                         event_rate_target_reached=stop)
        proposal.adaptation_outcome['updates_completed'] = proposal.update_steps
        if step == config['max_rounds'] and not stop:
            proposal.adaptation_outcome['status'] = 'round_limit_reached'
        row = metric_row(batch, step, proposal.last_fit_metrics, fit_seconds)
        row['proposal_metrics'] = accuracy.prepare_ce_metrics(row['proposal_metrics'], batch['event'])
        row.update(accuracy.record_estimate(step, row['is_estimate'], row['log_is_estimate']))
        if hasattr(proposal.tokenizer, 'batch_decode'):
            row['sample_texts'] = proposal.tokenizer.batch_decode(row['sample_token_ids'])
            row['event_sample_texts'] = proposal.tokenizer.batch_decode(row['event_sample_token_ids'])
        train_rows.append(row)
        progress = dict(phase='train', hits=hits, **{k:v for k,v in row.items()
                        if k not in ('event_indicator','log_importance_weights','log_p_seq','log_q_seq')})
        with (output/'progress.jsonl').open('a') as handle:
            handle.write(json.dumps(progress)+'\n')
        accuracy.print_ce_iteration(row, batch['ids'][:, batch['prefix_len']:], batch['event'])
        if step == 1 or step % config.get('checkpoint_every',10) == 0 or stop or step == config['max_rounds']:
            save_metrics(output/'metrics_atomic.pkl', train_rows, metadata)
            proposal.save(str(output))
        del batch
        if stop:
            print(reason, flush=True)
            break
    proposal.freeze_for_eval()
    evaluation = []
    accuracy = accuracy_logger('eval')
    for step in range(1, config['eval_batches']+1):
        batch = collect_batch(proposal, reference, event, prefix, config['eval_batch_size'],
                              config['length'], config['microbatch'])
        row = metric_row(batch, step, {}, 0.)
        row.update(accuracy.record_estimate(step, row['is_estimate'], row['log_is_estimate']))
        evaluation.append(row)
        with (output/'eval_progress.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(phase='eval', **{k:v for k,v in row.items()
                         if k not in ('event_indicator','log_importance_weights','log_p_seq','log_q_seq')}))+'\n')
        if step == 1 or step % 10 == 0 or step == config['eval_batches']:
            save_metrics(output/'eval/metrics_atomic.pkl', evaluation,
                         dict(metadata, phase='eval', batch_size=config['eval_batch_size'],
                              training_batch_size=config['batch_size']))
        accuracy.print_ce_iteration(row, batch['ids'][:, batch['prefix_len']:], batch['event'])
        del batch
    costs = dict(train_seconds=sum(sum(r['timings'].values()) for r in train_rows),
                 evaluation_seconds=sum(sum(r['timings'].values()) for r in evaluation),
                 fit_likelihood_evaluations=proposal.total_fit_likelihood_evaluations,
                 gradient_steps=sum(r['proposal_metrics'].get('gradient_steps',0) for r in train_rows),
                 gradient_backward_trajectories=sum(r['proposal_metrics'].get('gradient_backward_trajectories',0)
                                                   for r in train_rows),
                 elite_reference_recompute_count=sum(r['proposal_metrics'].get('elite_count',0) for r in train_rows)
                    if proposal.MODE == 'logit' else 0,
                 train_trajectories=len(train_rows)*config['batch_size'],
                 evaluation_trajectories=len(evaluation)*config['eval_batch_size'])
    (output/'compute_costs.json').write_text(json.dumps(costs, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    case = next(case for case in config.pop('cases') if case['name'] == args.case)
    config.update(case)
    for key in ('batch_size', 'length', 'microbatch', 'max_rounds', 'eval_batches', 'eval_batch_size'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config.get('checkpoint_every', 10) < 1:
        raise ValueError('checkpoint_every must be positive')
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    (args.output/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    device = torch.device('cuda')
    tokenizer = _load_tokenizer(config['model'])
    event = TokenEvent.from_config(tokenizer, dict(token=config['token'], gt_prob=config['gt_prob'], surrogate='sum'))
    model = _load_causal_lm(config['model']).eval()
    reference = copy.deepcopy(model).to(device).eval()
    reference.requires_grad_(False)
    if config['mode'] == 'lora':
        model = _setup_lora_model(model, SimpleNamespace(
            use_lora=True, lora_rank=config['lora_rank'], lora_alpha=config['lora_alpha'],
            lora_dropout=0.0))
        proposal = CEMLELoRAProposal(
            model=model, tokenizer=tokenizer, device=device, model_source=config['model'],
            elite_ratio=config['elite_ratio'], score_batch_size=config['score_batch_size'],
            fit_learning_rate=config['fit_learning_rate'], fit_steps=config['fit_steps'],
            stop_event_rate=config['event_rate_target'], max_rounds=config['max_rounds'],
            importance_weighted=config.get('importance_weighted', True))
    elif config['mode'] in ('activation', 'logit'):
        cls = CEMLEActivationProposal if config['mode'] == 'activation' else CEMLELogitProposal
        proposal = cls(base_model=model, tokenizer=tokenizer, device=device, model_source=config['model'],
                   elite_ratio=config['elite_ratio'], score_batch_size=config['score_batch_size'],
                   fit_method='gradient', importance_weighted=config.get('importance_weighted', True),
                   fit_learning_rate=config.get('fit_learning_rate', 0.01),
                   fit_steps=config.get('fit_steps', 1),
                   score_mode='surrogate', stop_event_rate=config['event_rate_target'],
                   max_rounds=config['max_rounds'], early_stop_on_low_ess=False)
    else:
        raise ValueError(f"Unknown CE proposal mode: {config['mode']}")
    prefix = proposal.encode_context(config['context'])
    print(f"Loaded {config['mode']} model={config['model']} token={config['token']!r} "
          f"id={event.token_id} batch={config['batch_size']} weighted={proposal.importance_weighted} "
          f"fit={proposal.fit_method} fit_steps={proposal.fit_steps} fit_lr={proposal.fit_learning_rate} "
          f"max_rounds={config['max_rounds']} microbatch={config['microbatch']}", flush=True)
    run(config, args.output, proposal, reference, event, prefix)


if __name__ == '__main__':
    main()
