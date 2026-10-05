"""One source-trained ResidualTCN shared over 1s, 2s, and 4s inputs."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .phase3 import set_determinism
from .r2_data import FACEDWindowIndex, GuardedRaw
from .r2_models import mean_subject_bacc
from .r3_core import load_anchor_batch
from .r3_training import predict_r3
from .types import FoldSubjects


CONTEXTS = ("1s", "2s", "4s")


def train_shared_encoder(model, raw_train: GuardedRaw, raw_validation: GuardedRaw,
                         index: FACEDWindowIndex, anchors: np.ndarray,
                         validation: np.ndarray, contexts: dict[str, np.ndarray],
                         normalizer, fold: FoldSubjects, config: dict, seed: int,
                         device: torch.device) -> tuple[dict, dict]:
    if (set(index.subject_ids[anchors]) != set(fold.source_train_subjects) or
            set(index.subject_ids[validation]) != set(fold.source_validation_subjects) or
            fold.held_out_subject in index.subject_ids[np.r_[anchors, validation]] or
            set(normalizer.fit_subjects) != set(fold.source_train_subjects)):
        raise AssertionError("R5 shared encoder fitting or selection includes target data")
    settings = config["shared_weight_control"]
    set_determinism(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["encoder_learning_rate"],
                                  weight_decay=config["head_training"]["weight_decay"])
    history, best_state, best_score, best_epoch, stale = [], None, -np.inf, 0, 0
    for epoch in range(1, settings["encoder_maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(anchors), generator=torch.Generator().manual_seed(seed + epoch)).numpy()
        loss_total = 0.0
        for batch_number, start in enumerate(range(0, len(order), settings["encoder_batch_size"])):
            rows = anchors[order[start:start + settings["encoder_batch_size"]]]
            context_name = CONTEXTS[(batch_number + epoch - 1) % len(CONTEXTS)]
            block = load_anchor_batch(raw_train, index, rows, contexts[context_name],
                                      set(fold.source_train_subjects), normalizer)
            logits, _ = model(torch.from_numpy(block).to(device))
            loss = F.cross_entropy(logits, torch.as_tensor(index.y[rows], dtype=torch.long,
                                                          device=device))
            if not torch.isfinite(loss):
                raise FloatingPointError("R5 shared encoder source loss nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config["head_training"]["gradient_clip_norm"],
                                           error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach()) * len(rows)
        scores = {}
        for context_name in CONTEXTS:
            probabilities, _, _ = predict_r3(
                model, raw_validation, index, validation, contexts[context_name],
                set(fold.source_validation_subjects), normalizer, device,
                settings["encoder_batch_size"])
            scores[context_name] = mean_subject_bacc(
                index.y[validation], probabilities, index.subject_ids[validation])
        score = float(np.mean(list(scores.values())))
        history.append({"epoch": epoch, "source_train_mixed_context_loss": loss_total / len(anchors),
                        "source_validation_bacc_by_context": scores,
                        "source_validation_mean_bacc_across_contexts": score})
        if score > best_score + 1e-8:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= settings["encoder_patience"]:
                break
    if best_state is None:
        raise RuntimeError("R5 shared encoder has no source-validation-selected checkpoint")
    return best_state, {"seed": seed, "history": history, "best_epoch": best_epoch,
                        "best_source_validation_mean_bacc_across_contexts": best_score,
                        "source_training_subjects": list(fold.source_train_subjects),
                        "source_validation_subjects": list(fold.source_validation_subjects),
                        "target_used_for_training_or_selection": False}
