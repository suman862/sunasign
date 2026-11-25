#training
import argparse
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
import yaml

from .data import NSLDataset, Vocabulary, collate_fn, load_jsonl
from .losses import compute_loss
from .model import SunaSign


def set_seed(s):
  random.seed(s)
  torch.manual_seed(s)
  torch.cuda.manual_seed_all(s)


def build_vocab(rows):
  toks = []
  for r in rows:
    toks.extend(r["text"].split())
  return Vocabulary(toks)


def run(args):
  cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
  set_seed(cfg["seed"])
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}", flush=True)

  train_rows = load_jsonl(cfg["data"]["train_jsonl"])
  val_rows = load_jsonl(cfg["data"]["val_jsonl"])
  print(
      f"Loaded {len(train_rows)} raw train rows and {len(val_rows)} val rows.",
      flush=True,
  )

  vocab = build_vocab(train_rows + val_rows)
  cfg["model"]["vocab_size"] = len(vocab.itos)
  print(f"Vocabulary size: {len(vocab.itos)}", flush=True)

  train_ds = NSLDataset(
      train_rows,
      cfg["data"]["root"],
      cfg["data"]["max_frames"],
      cfg["data"]["image_size"],
      cfg["data"]["landmark_dim"],
      vocab,
  )
  val_ds = NSLDataset(
      val_rows,
      cfg["data"]["root"],
      cfg["data"]["max_frames"],
      cfg["data"]["image_size"],
      cfg["data"]["landmark_dim"],
      vocab,
  )
  print(
      f"Dataset active samples - Train: {len(train_ds)}, Val: {len(val_ds)}",
      flush=True,
  )

  if len(train_ds) == 0:
    raise ValueError(
        "Train dataset is empty! Check video file paths in train.jsonl."
    )

  train_dl = DataLoader(
      train_ds,
      batch_size=cfg["train"]["batch_size"],
      shuffle=True,
      num_workers=cfg["data"]["num_workers"],
      collate_fn=collate_fn,
      pin_memory=True,
  )
  val_dl = DataLoader(
      val_ds,
      batch_size=cfg["train"]["batch_size"],
      shuffle=False,
      num_workers=cfg["data"]["num_workers"],
      collate_fn=collate_fn,
      pin_memory=True,
  )
  print(
      f"DataLoader batches - Train: {len(train_dl)}, Val:"
      f" {len(val_dl)}\nStarting training...",
      flush=True,
  )

  model = SunaSign(cfg["model"]).to(device)
  opt = torch.optim.AdamW(
      model.parameters(),
      lr=cfg["train"]["lr"],
      weight_decay=cfg["train"]["weight_decay"],
  )
  scaler = torch.amp.GradScaler(
      "cuda", enabled=(cfg["train"]["amp"] and device.type == "cuda")
  )
  best = float("inf")
  Path(cfg["train"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

  epochs = cfg["train"]["epochs"]
  for epoch in range(epochs):
    model.train()
    total = 0.0
    for step, (frames, lm, tokens, boundary, pad) in enumerate(train_dl):
      frames = frames.to(device, non_blocking=True).float() / 255.0
      lm = lm.to(device, non_blocking=True)
      tokens = tokens.to(device, non_blocking=True)
      boundary = boundary.to(device, non_blocking=True)
      pad = pad.to(device, non_blocking=True)

      opt.zero_grad(set_to_none=True)
      with torch.autocast(
          device_type=device.type,
          enabled=(device.type == "cuda" and cfg["train"]["amp"]),
      ):
        out = model(frames, lm, tokens, pad)
        loss, _ = compute_loss(out, tokens, boundary, pad)

      scaler.scale(loss).backward()
      scaler.unscale_(opt)
      torch.nn.utils.clip_grad_norm_(
          model.parameters(), cfg["train"]["grad_clip"]
      )
      scaler.step(opt)
      scaler.update()

      total += loss.item()

    model.eval()
    vtotal = 0.0
    with torch.no_grad():
      for frames, lm, tokens, boundary, pad in val_dl:
        frames = frames.to(device, non_blocking=True).float() / 255.0
        lm = lm.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        boundary = boundary.to(device, non_blocking=True)
        pad = pad.to(device, non_blocking=True)

        out = model(frames, lm, tokens, pad)
        loss, _ = compute_loss(out, tokens, boundary, pad)
        vtotal += loss.item()

    train_loss = total / max(1, len(train_dl))
    val_loss = vtotal / max(1, len(val_dl))
    print(
        f"epoch={epoch+1}/{epochs} train_loss={train_loss:.4f}"
        f" val_loss={val_loss:.4f}",
        flush=True,
    )

    if val_loss < best:
      best = val_loss
      torch.save(
          {
              "model": model.state_dict(),
              "config": cfg,
              "vocab": vocab.itos,
          },
          Path(cfg["train"]["checkpoint_dir"]) / "best.pt",
      )


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--config", default="configs/base.yaml")
  run(p.parse_args())