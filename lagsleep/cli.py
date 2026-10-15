import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

from .method import LagSleep

OFFSETS = np.array([-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5])


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    partitions, packed = {}, {}
    for row in rows:
        subject, split = str(row["subject"]), row["split"]
        if subject in partitions and partitions[subject] != split:
            raise ValueError("All nights of a subject must remain in the same split")
        partitions[subject] = split
        data = np.load(manifest.resolve().parent / row["file"], allow_pickle=False)
        fs = float(data["fs"])
        if fs != args.fs:
            raise ValueError("Resample both channels to the requested frequency before preparation")
        eeg, eog = data["eeg"].reshape(-1), data["eog"].reshape(-1)
        labels = data["labels"].astype(int)
        if len(eeg) != len(eog):
            raise ValueError("EEG and EOG must have a common timestamp origin and length")
        margin, length = round(10 * fs), round(30 * fs)
        transition = np.zeros(len(labels), bool)
        for k in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            transition[max(0, k - args.transition_radius):min(len(labels), k + args.transition_radius + 1)] = True
        for k, label in enumerate(labels):
            start, end = k * length, (k + 1) * length
            if label not in range(5) or start < margin or end + margin > len(eog):
                continue
            e = eeg[start:end].astype(np.float32)
            o = eog[start - margin:end + margin].astype(np.float32)
            if not np.isfinite(e).all():
                continue
            usable = bool(np.isfinite(o).all() and np.std(o) > 1e-6)
            o = np.nan_to_num(o, nan=0, posinf=0, neginf=0)
            correlations = []
            for offset in OFFSETS:
                lo = margin + round(offset * fs)
                correlations.append(abs(np.corrcoef(e, o[lo:lo + length])[0, 1]))
            scores = np.sort(np.nan_to_num(correlations))
            identifiable = bool(usable and transition[k] and scores[-1] >= 0.25 and scores[-1] - scores[-2] >= 0.02)
            packed.setdefault(split, []).append((subject, e, o, label, transition[k], identifiable, usable))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in packed.items():
        subject, eeg, eog, y, transition, identifiable, eligible = zip(*samples)
        np.savez_compressed(out / f"{split}.npz", subject=subject, eeg=np.array(eeg), eog=np.array(eog),
                            y=y, transition=transition, identifiable=identifiable, eligible=eligible, fs=args.fs)


def batch(data, index, shift, device):
    fs = float(data["fs"])
    eeg = data["eeg"][index]
    eog = data["eog"][index]
    shift = np.broadcast_to(np.asarray(shift), (len(index),))
    candidates = np.stack([np.stack([o[round((10 + delta + s) * fs):round((10 + delta + s) * fs) + eeg.shape[1]] for delta in OFFSETS]) for o, s in zip(eog, shift)])
    return (torch.as_tensor(eeg, device=device), torch.as_tensor(candidates, device=device),
            torch.as_tensor(data["y"][index], dtype=torch.long, device=device))


def choose(output, method, available, js_threshold, entropy_threshold):
    if method == "eeg":
        return output["eeg"], torch.ones_like(available, dtype=torch.bool)
    if method == "hard":
        prediction = output["candidates"][torch.arange(len(available), device=available.device), output["weights"].argmax(1)]
    elif method == "feature":
        prediction = output["feature"]
    else:
        prediction = output["mixture"]
    fallback = ~available
    if method == "full":
        fallback = fallback | (output["js"] > js_threshold)
    if method == "entropy":
        fallback = fallback | (output["offset_entropy"] > entropy_threshold)
    return torch.where(fallback[:, None], output["eeg"], prediction), fallback


def infer(net, data, args, method, shift, missing, threshold=0, entropy_threshold=0):
    predictions, js, entropies, fallbacks, mixtures, unis = [], [], [], [], [], []
    rng = np.random.default_rng(args.seed)
    with torch.no_grad():
        for start in range(0, len(data["y"]), args.batch_size):
            index = np.arange(start, min(start + args.batch_size, len(data["y"])))
            e, o, _ = batch(data, index, shift, args.device)
            out = net(e, o)
            available = torch.as_tensor((rng.random(len(index)) >= missing) & data["eligible"][index], device=args.device)
            prediction, fallback = choose(out, method, available, threshold, entropy_threshold)
            predictions.append(prediction.cpu().numpy())
            js.append(out["js"].cpu().numpy())
            entropies.append(out["offset_entropy"].cpu().numpy())
            fallbacks.append(fallback.cpu().numpy())
            mixtures.append(out["mixture"].cpu().numpy())
            unis.append(out["eeg"].cpu().numpy())
    return tuple(np.concatenate(values) for values in [predictions, js, entropies, fallbacks, mixtures, unis])


def macro(y, p):
    return float(f1_score(y, p, labels=list(range(5)), average="macro", zero_division=0))


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data, dev = [np.load(Path(args.data) / f"{split}.npz", allow_pickle=False) for split in ["train", "dev"]]
    if set(data["subject"]) & set(dev["subject"]):
        raise ValueError("Development subjects overlap training subjects")
    net = LagSleep(args.width, args.temperature).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        net.train()
        order = rng.permutation(len(data["y"]))
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            shift = np.zeros(len(index)) if args.no_shift_augmentation else rng.choice(OFFSETS, len(index))
            e, o, y = batch(data, index, shift, args.device)
            out = net(e, o)
            available = torch.as_tensor((rng.random(len(index)) >= 0.2) & data["eligible"][index], device=args.device)
            if args.no_missing_loss:
                available = torch.ones_like(available)
            probability = torch.where(available[:, None], out["mixture"], out["eeg"])
            loss = F.nll_loss(probability.clamp_min(1e-8).log(), y)
            if not args.no_missing_loss:
                loss = loss + F.nll_loss(out["eeg"].clamp_min(1e-8).log(), y)
            mask = torch.as_tensor(data["identifiable"][index], dtype=torch.bool, device=args.device)
            if mask.any() and not args.no_shift_loss:
                target = torch.as_tensor([np.argmin(abs(OFFSETS + s)) for s in shift], device=args.device)
                loss = loss + 0.2 * F.nll_loss(out["weights"][mask].clamp_min(1e-8).log(), target[mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    net.eval()
    _, js, ent, _, mixture, uni = infer(net, dev, args, "mixture", args.dev_shift, 0)
    thresholds = []
    for scores in [js, ent]:
        grid = np.unique(np.r_[-1e-8, np.quantile(scores, np.linspace(0, 1, 101)), scores.max() + 1e-8])
        values = [macro(dev["y"], np.where((scores > t)[:, None], uni, mixture).argmax(1)) for t in grid]
        thresholds.append(float(grid[np.argmax(values)]))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "width": args.width, "temperature": args.temperature,
                "threshold": thresholds[0], "entropy_threshold": thresholds[1],
                "fs": float(data["fs"]),
                "subjects": np.unique(np.r_[data["subject"], dev["subject"]]).tolist()}, out / "model.pt")


def evaluate(args):
    data = np.load(Path(args.data) / "test.npz", allow_pickle=False)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if set(saved["subjects"]) & set(data["subject"]):
        raise ValueError("Test subjects overlap training/development subjects")
    if float(data["fs"]) != saved["fs"]:
        raise ValueError("Test sampling frequency differs from fitting")
    net = LagSleep(saved["width"], saved["temperature"]).to(args.device)
    net.load_state_dict(saved["model"])
    net.eval()
    summaries, probability = [], {}
    for shift in args.shifts:
        for missing in args.missing:
            p, js, entropy, fallback, _, _ = infer(net, data, args, args.method, shift, missing, saved["threshold"], saved["entropy_threshold"])
            pred = p.argmax(1)
            transition = data["transition"].astype(bool)
            summaries.append({"offset_seconds": shift, "missing_fraction": missing,
                              "macro_f1": macro(data["y"], pred),
                              "transition_macro_f1": macro(data["y"][transition], pred[transition]) if transition.any() else None,
                              "fallback_fraction": float(fallback.mean())})
            key = f"shift_{shift}_missing_{missing}"
            probability[key] = p
            probability[key + "_js"] = js
            probability[key + "_entropy"] = entropy
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    np.savez_compressed(out / "predictions.npz", y=data["y"], subject=data["subject"], **probability)
    cm = confusion_matrix(data["y"], next(iter(probability.values())).argmax(1), labels=range(5))
    np.savetxt(out / "confusion.csv", cm, delimiter=",", fmt="%d")
    fig, ax = plt.subplots(figsize=(6, 4))
    for missing in args.missing:
        rows = [r for r in summaries if r["missing_fraction"] == missing]
        ax.plot([r["offset_seconds"] for r in rows], [r["transition_macro_f1"] for r in rows], marker="o", label=f"Missing {missing:g}")
    ax.set(xlabel="EOG offset (seconds)", ylabel="Transition macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "offset_curve.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--fs", type=float, default=100)
    p.add_argument("--transition-radius", type=int, default=1)
    p.set_defaults(function=prepare)
    for name in ["train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--seed", type=int, default=42)
        if name == "train":
            p.add_argument("--epochs", type=int, default=40)
            p.add_argument("--width", type=int, default=48)
            p.add_argument("--temperature", type=float, default=1)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--dev-shift", type=float, default=3)
            for flag in ["no-missing-loss", "no-shift-loss", "no-shift-augmentation"]:
                p.add_argument("--" + flag, action="store_true")
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--method", choices=["full", "eeg", "hard", "feature", "entropy", "mixture"], default="full")
            p.add_argument("--shifts", type=float, nargs="+", default=[0, 0.5, 1, 2, 3, 5])
            p.add_argument("--missing", type=float, nargs="+", default=[0])
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    if hasattr(args, "shifts") and any(abs(s) > 5 for s in args.shifts):
        parser.error("Offsets must remain within the declared +/-5 second range")
    args.function(args)
