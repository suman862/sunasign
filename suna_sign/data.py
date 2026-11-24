import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Vocabulary:

  def __init__(self, tokens):
    self.pad = "<pad>"
    self.bos = "<bos>"
    self.eos = "<eos>"
    self.unk = "<unk>"
    self.itos = [self.pad, self.bos, self.eos, self.unk] + sorted(
        set(tokens)
    )
    self.stoi = {x: i for i, x in enumerate(self.itos)}

    self.pad_idx = self.stoi[self.pad]
    self.bos_idx = self.stoi[self.bos]
    self.eos_idx = self.stoi[self.eos]
    self.unk_idx = self.stoi[self.unk]

  def encode(self, text):
    words = text.strip().split()
    return [self.stoi.get(w, self.unk_idx) for w in words]

  def __len__(self):
    return len(self.itos)


def load_jsonl(path):
  with open(path, "r", encoding="utf-8") as f:
    content = f.read().strip()

  try:
    return [json.loads(line) for line in content.splitlines() if line.strip()]
  except json.JSONDecodeError:
    # Fallback for concatenated or multiline JSON arrays
    return json.loads("[" + content.replace("}\n{", "},\n{") + "]")


class NSLDataset(Dataset):

  def __init__(self, rows, root, max_frames, image_size, landmark_dim, vocab):
    self.rows = rows
    self.root = Path(root)
    self.max_frames = max_frames
    self.image_size = image_size
    self.landmark_dim = landmark_dim
    self.vocab = vocab

  def __len__(self):
    return len(self.rows)

  def _read_video(self, path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
      raise RuntimeError(f"Cannot open video file: {path}")

    frames = []
    while len(frames) < self.max_frames:
      ok, frame = cap.read()
      if not ok:
        break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame = cv2.resize(frame, (self.image_size, self.image_size))
      frames.append(frame)
    cap.release()

    if not frames:
      raise RuntimeError(f"Video file contains no readable frames: {path}")

    # Memory optimization: Return uint8 ByteTensor [T, C, H, W]
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2)

  def __getitem__(self, i):
    r = self.rows[i]
    video = self._read_video(self.root / r["video"])
    t = video.size(0)

    # Align Landmark features with decoded video length
    lm = np.zeros((t, self.landmark_dim), dtype=np.float32)
    if "landmarks" in r:
      lm_path = self.root / r["landmarks"]
      if lm_path.exists():
        lm_data = np.load(lm_path).astype(np.float32)
        if lm_data.ndim == 2 and lm_data.shape[1] == self.landmark_dim:
          min_t = min(t, len(lm_data))
          lm[:min_t] = lm_data[:min_t]

    # Encode token sequence with explicit BOS/EOS markers
    tokens = [self.vocab.bos_idx] + self.vocab.encode(r["text"]) + [self.vocab.eos_idx]
    token_tensor = torch.tensor(tokens, dtype=torch.long)

    # Frame boundary target setup
    boundary = torch.zeros(t, dtype=torch.float32)
    for a, b in r.get("segments", []):
      if 0 <= a < t:
        boundary[a] = 1.0
      if 0 <= b < t:
        boundary[b] = 1.0

    return {
        "frames": video,  # [T, C, H, W] uint8
        "landmarks": torch.from_numpy(lm),  # [T, D] float32
        "tokens": token_tensor,  # [L] long
        "boundary": boundary,  # [T] float32
        "pad_idx": self.vocab.pad_idx,
    }


def collate_fn(batch):
  b = len(batch)
  max_t = max(x["frames"].size(0) for x in batch)
  max_l = max(x["tokens"].size(0) for x in batch)
  c, h, w = batch[0]["frames"].shape[1:]
  d = batch[0]["landmarks"].size(1)
  pad_idx = batch[0]["pad_idx"]

  frames = torch.zeros(b, max_t, c, h, w, dtype=torch.uint8)
  landmarks = torch.zeros(b, max_t, d, dtype=torch.float32)
  boundary = torch.zeros(b, max_t, dtype=torch.float32)
  pad = torch.ones(b, max_t, dtype=torch.bool)  # True indicates padding frame
  tokens = torch.full((b, max_l), fill_value=pad_idx, dtype=torch.long)

  for i, item in enumerate(batch):
    ti = item["frames"].size(0)
    li = item["tokens"].size(0)

    frames[i, :ti] = item["frames"]
    landmarks[i, :ti] = item["landmarks"]
    boundary[i, :ti] = item["boundary"]
    pad[i, :ti] = False
    tokens[i, :li] = item["tokens"]

  return frames, landmarks, tokens, boundary, pad