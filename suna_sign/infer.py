import argparse, yaml, cv2, numpy as np, torch
from .model import SunaSign


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    vocab = ckpt["vocab"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SunaSign(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("cannot open camera")

    frames = []
    print("Press q to quit. This baseline expects landmark vectors; zeros are used until a landmark extractor is integrated.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (cfg["data"]["image_size"], cfg["data"]["image_size"]))
        frames.append(torch.from_numpy(rgb).permute(2,0,1).float()/255.0)
        frames = frames[-cfg["data"]["max_frames"]:]
        cv2.imshow("SunaSign", frame)
        if len(frames) >= 8:
            x = torch.stack(frames)[None].to(device)
            lm = torch.zeros(1, x.size(1), cfg["data"]["landmark_dim"], device=device)
            pad = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
            ids = model.generate_greedy(
                x, lm, 1, 2, 0, cfg["model"]["max_text_len"], pad
            )[0].tolist()
            words = [vocab[i] for i in ids[1:] if i not in (0,2)]
            print("\rPrediction:", " ".join(words), end="", flush=True)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera", type=int, default=0)
    main(p.parse_args())
