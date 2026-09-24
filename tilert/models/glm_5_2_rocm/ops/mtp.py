"""GLM-5.2 MTP step-tail helpers: the CPU goldens for the fused tail."""

import torch

HIDDEN = 6144
SIMULATE_TOKEN_ID = 100


def xfer_buf_bytes(producer_samples: int, topk: int) -> int:
    return int(torch.ops.tilert.glm5_broadcast_xfer_buf_bytes(producer_samples, topk))


def verify_golden(
    draft_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    mtp0_tokens: torch.Tensor,
    mtp0_hidden: torch.Tensor,
    cur_pos: torch.Tensor,
    idx_source: torch.Tensor | None = None,
    sim_accept: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    d = draft_tokens.cpu()
    p = predicted_tokens.cpu().clone()
    m = mtp0_tokens.cpu().clone()
    h = mtp0_hidden.cpu()
    batch, mtp_seq_len = d.shape
    num_verified = mtp_seq_len - 1
    out = {
        "num_accepted": torch.zeros(batch, dtype=torch.int32),
        "cur_pos": cur_pos.cpu().clone(),
        "last_token": torch.zeros(batch, dtype=torch.int32),
        "last_hidden": torch.zeros(batch, HIDDEN, dtype=h.dtype),
        "next_draft_tokens": torch.zeros(batch, mtp_seq_len, dtype=torch.int32),
    }
    if idx_source is not None:
        topk = idx_source.shape[1]
        out["idx_selects"] = torch.zeros(batch, topk, dtype=torch.int32)
    for b in range(batch):
        if sim_accept is not None:
            acc = min(max(int(sim_accept.cpu()[b]), 0), mtp_seq_len - 1)
            p[b, :] = SIMULATE_TOKEN_ID
            m[b, :] = SIMULATE_TOKEN_ID
        else:
            acc = 0
            for i in range(num_verified):
                if d[b, i + 1] == p[b, i]:
                    acc += 1
                else:
                    break
        out["num_accepted"][b] = acc + 1
        out["cur_pos"][b] += acc + 1
        out["last_token"][b] = m[b, acc]
        out["next_draft_tokens"][b, 0] = p[b, acc]
        out["next_draft_tokens"][b, 1] = m[b, acc]
        out["last_hidden"][b] = h[b * mtp_seq_len + acc]
        if idx_source is not None:
            out["idx_selects"][b] = idx_source.cpu()[b * mtp_seq_len + acc]
    out["predicted_tokens"] = p
    out["mtp0_tokens"] = m
    return out


def assemble_accepted_golden(
    num_accepted: torch.Tensor, predicted: torch.Tensor, ar_acc: torch.Tensor, ar_num: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    acc = ar_acc.cpu().clone()
    num = ar_num.cpu().clone()
    na = num_accepted.cpu()
    pred = predicted.cpu()
    acc_stride = acc.shape[1]
    num_stride = num.shape[1]
    for b in range(pred.shape[0]):
        n = int(na[b])
        base = int(acc[b, 0]) + 1
        for i in range(n):
            if base + i < acc_stride:
                acc[b, base + i] = pred[b, i]
        acc[b, 0] = min(int(acc[b, 0]) + n, acc_stride - 1)
        k = int(num[b, 0])
        if 1 + k < num_stride:
            num[b, 1 + k] = n
        num[b, 0] = min(k + 1, num_stride - 1)
    return (acc, num)


def step_tail_golden(
    draft_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    mtp0_tokens: torch.Tensor,
    mtp0_hidden: torch.Tensor,
    cur_pos: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    idx_selects: torch.Tensor,
    ar_acc: torch.Tensor,
    ar_num: torch.Tensor,
    idx_source: torch.Tensor | None = None,
    local_gather: bool = False,
    sim_accept: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch, mtp_seq_len = draft_tokens.shape
    src = None
    if idx_source is not None:
        src = idx_source
    elif local_gather:
        src = idx_selects[: batch * mtp_seq_len]
    v = verify_golden(
        draft_tokens, predicted_tokens, mtp0_tokens, mtp0_hidden, cur_pos, src, sim_accept
    )
    acc, num = assemble_accepted_golden(v["num_accepted"], v["predicted_tokens"], ar_acc, ar_num)
    positions = v["cur_pos"].long()[:, None] + torch.arange(mtp_seq_len)[None, :]
    rotated = next_draft_tokens.cpu().clone()
    rotated[:, :2] = v["next_draft_tokens"][:, :2]
    idx = idx_selects.cpu().clone()
    if src is not None:
        idx[:batch] = v["idx_selects"]
    return {
        "draft_tokens": rotated.clone(),
        "predicted_tokens": v["predicted_tokens"],
        "mtp0_tokens": v["mtp0_tokens"],
        "num_accepted": v["num_accepted"],
        "cur_pos": v["cur_pos"],
        "last_token": v["last_token"],
        "last_hidden": v["last_hidden"],
        "next_draft_tokens": rotated,
        "idx_selects": idx,
        "ar_acc": acc,
        "ar_num": num,
        "positions": positions,
    }
