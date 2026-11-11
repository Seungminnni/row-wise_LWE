from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plot_relation_pair_tsne_binary_secret import (  # noqa: E402
    balanced_subset,
    collect_coordinate_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rotatable 3D HTML t-SNE for relation_pair secret-bit embeddings.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs" / "n16_h2_noise3_m64_relation_pair_cnn1d_10k_1k_1k_ep200_seed0",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "result")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tsne-seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-points-per-class", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def tsne_3d(x: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    x = StandardScaler().fit_transform(x)
    points = TSNE(
        n_components=3,
        init="pca",
        learning_rate="auto",
        perplexity=min(perplexity, max(1.0, (x.shape[0] - 1) / 3.0)),
        random_state=seed,
        max_iter=max_iter,
    ).fit_transform(x)
    points -= points.mean(axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(points, axis=1), 95)
    if scale > 0:
        points /= scale
    return points.astype(float)


def save_coordinates(path: Path, before: np.ndarray, after: np.ndarray, y: np.ndarray, coords: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write("point,secret_bit,coordinate,before_x,before_y,before_z,after_x,after_y,after_z\n")
        for idx in range(len(y)):
            handle.write(
                f"{idx},{int(y[idx])},{int(coords[idx])},"
                f"{before[idx, 0]:.8f},{before[idx, 1]:.8f},{before[idx, 2]:.8f},"
                f"{after[idx, 0]:.8f},{after[idx, 1]:.8f},{after[idx, 2]:.8f}\n"
            )


def html_document(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    title = "Relation-pair 3D t-SNE by secret bit"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
      color: #172033;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0 0 18px;
      color: #546179;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: white;
      border: 1px solid #dce2ee;
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(32, 45, 72, 0.08);
      overflow: hidden;
    }}
    .panel-header {{
      padding: 12px 14px;
      border-bottom: 1px solid #e7ecf5;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .panel-title {{
      font-weight: 700;
      font-size: 15px;
    }}
    .hint {{
      color: #6b768b;
      font-size: 12px;
      white-space: nowrap;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 560px;
      cursor: grab;
      background: linear-gradient(#fbfcff, #f1f5fb);
    }}
    canvas:active {{
      cursor: grabbing;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      align-items: center;
      margin: 14px 0 0;
      font-size: 14px;
    }}
    .item {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
    }}
    .meta {{
      margin-top: 14px;
      color: #667187;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      body {{ padding: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
      canvas {{ height: 460px; }}
    }}
  </style>
</head>
<body>
  <h1>Relation-pair 3D t-SNE by secret bit</h1>
  <p class="sub">Coordinate-level embeddings colored by <code>s_j = 0</code> and <code>s_j = 1</code>. Drag to rotate, wheel to zoom, double-click to reset.</p>
  <div class="grid">
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">Before CNN</div>
        <div class="hint">flatten coordinate input: m x {payload['input_channels']}</div>
      </div>
      <canvas id="before"></canvas>
    </div>
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">After CNN</div>
        <div class="hint">flatten coordinate feature: m x {payload['hidden_channels']}</div>
      </div>
      <canvas id="after"></canvas>
    </div>
  </div>
  <div class="legend">
    <span class="item"><span class="dot" style="background:#3267c8"></span>s_j = 0</span>
    <span class="item"><span class="dot" style="background:#d33f49"></span>s_j = 1</span>
  </div>
  <div class="meta" id="meta"></div>
  <script>
    const DATA = {data};

    function multiplyRot(point, rx, ry) {{
      let [x, y, z] = point;
      const cy = Math.cos(ry), sy = Math.sin(ry);
      let x1 = cy * x + sy * z;
      let z1 = -sy * x + cy * z;
      const cx = Math.cos(rx), sx = Math.sin(rx);
      let y2 = cx * y - sx * z1;
      let z2 = sx * y + cx * z1;
      return [x1, y2, z2];
    }}

    class Viewer {{
      constructor(canvas, points, labels, coords) {{
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.points = points;
        this.labels = labels;
        this.coords = coords;
        this.rx = -0.42;
        this.ry = 0.62;
        this.zoom = 1.0;
        this.dragging = false;
        this.lastX = 0;
        this.lastY = 0;
        this.bind();
        this.resize();
      }}
      bind() {{
        window.addEventListener("resize", () => this.resize());
        this.canvas.addEventListener("pointerdown", (e) => {{
          this.dragging = true;
          this.canvas.setPointerCapture(e.pointerId);
          this.lastX = e.clientX;
          this.lastY = e.clientY;
        }});
        this.canvas.addEventListener("pointermove", (e) => {{
          if (!this.dragging) return;
          const dx = e.clientX - this.lastX;
          const dy = e.clientY - this.lastY;
          this.lastX = e.clientX;
          this.lastY = e.clientY;
          this.ry += dx * 0.008;
          this.rx += dy * 0.008;
          this.draw();
        }});
        this.canvas.addEventListener("pointerup", () => {{ this.dragging = false; }});
        this.canvas.addEventListener("pointercancel", () => {{ this.dragging = false; }});
        this.canvas.addEventListener("wheel", (e) => {{
          e.preventDefault();
          this.zoom *= Math.exp(-e.deltaY * 0.001);
          this.zoom = Math.max(0.35, Math.min(4.0, this.zoom));
          this.draw();
        }}, {{ passive: false }});
        this.canvas.addEventListener("dblclick", () => {{
          this.rx = -0.42;
          this.ry = 0.62;
          this.zoom = 1.0;
          this.draw();
        }});
      }}
      resize() {{
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        this.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.draw();
      }}
      drawAxes(cx, cy, scale) {{
        const axes = [
          [[1.25,0,0], "#9aa7bb", "x"],
          [[0,1.25,0], "#9aa7bb", "y"],
          [[0,0,1.25], "#9aa7bb", "z"],
        ];
        const ctx = this.ctx;
        ctx.save();
        ctx.lineWidth = 1;
        ctx.font = "12px sans-serif";
        for (const [p, color, label] of axes) {{
          const r = multiplyRot(p, this.rx, this.ry);
          const depth = 3.5 / (3.5 - r[2]);
          const x = cx + r[0] * scale * depth;
          const y = cy - r[1] * scale * depth;
          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(x, y);
          ctx.stroke();
          ctx.fillStyle = color;
          ctx.fillText(label, x + 4, y + 4);
        }}
        ctx.restore();
      }}
      draw() {{
        const rect = this.canvas.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, w, h);
        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(w, h) * 0.36 * this.zoom;
        this.drawAxes(cx, cy, scale);

        const projected = this.points.map((p, i) => {{
          const r = multiplyRot(p, this.rx, this.ry);
          const depth = 3.5 / (3.5 - r[2]);
          return {{
            x: cx + r[0] * scale * depth,
            y: cy - r[1] * scale * depth,
            z: r[2],
            label: this.labels[i],
            coord: this.coords[i],
            depth
          }};
        }}).sort((a, b) => a.z - b.z);

        for (const p of projected) {{
          const isOne = p.label === 1;
          ctx.beginPath();
          ctx.fillStyle = isOne ? "rgba(211,63,73,0.82)" : "rgba(50,103,200,0.58)";
          ctx.arc(p.x, p.y, isOne ? 4.2 : 3.2, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
    }}

    const beforeViewer = new Viewer(document.getElementById("before"), DATA.before, DATA.labels, DATA.coords);
    const afterViewer = new Viewer(document.getElementById("after"), DATA.after, DATA.labels, DATA.coords);
    document.getElementById("meta").textContent =
      `points: ${{DATA.labels.length}} | s_j=0: ${{DATA.counts[0]}} | s_j=1: ${{DATA.counts[1]}} | ` +
      `t-SNE: 3D, perplexity=${{DATA.meta.perplexity}}, max_iter=${{DATA.meta.max_iter}}`;
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    before, after, y, coords, shapes = collect_coordinate_embeddings(args, device)
    keep = balanced_subset(y, args.max_points_per_class, args.seed)
    before = before[keep]
    after = after[keep]
    y = y[keep]
    coords = coords[keep]

    print("relation_pair coordinate tensor shapes:", json.dumps({k: list(v) for k, v in shapes.items()}, indent=2), flush=True)
    print(f"balanced points: zeros={(y == 0).sum()}, ones={(y == 1).sum()}", flush=True)
    print(f"fitting before-CNN 3D t-SNE: {before.shape}", flush=True)
    before_3d = tsne_3d(before, args.tsne_seed, args.perplexity, args.max_iter)
    print(f"fitting after-CNN 3D t-SNE: {after.shape}", flush=True)
    after_3d = tsne_3d(after, args.tsne_seed, args.perplexity, args.max_iter)

    html_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn_3d.html"
    csv_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn_3d.csv"
    meta_path = args.output_dir / "relation_pair_tsne_secret_bit_0_1_before_after_cnn_3d_meta.json"

    payload = {
        "input_channels": shapes["cnn_input_row_tensor"][1],
        "hidden_channels": shapes["cnn_output_row_tensor"][1],
        "before": before_3d.round(6).tolist(),
        "after": after_3d.round(6).tolist(),
        "labels": y.astype(int).tolist(),
        "coords": coords.astype(int).tolist(),
        "counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "meta": {"perplexity": args.perplexity, "max_iter": args.max_iter},
    }
    html_path.write_text(html_document(payload))
    save_coordinates(csv_path, before_3d, after_3d, y, coords)
    meta_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "num_samples": args.num_samples,
                "seed": args.seed,
                "tsne_seed": args.tsne_seed,
                "perplexity": args.perplexity,
                "max_iter": args.max_iter,
                "max_points_per_class": args.max_points_per_class,
                "class_counts_after_balance": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
                "shapes": {key: list(value) for key, value in shapes.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"saved {html_path}")
    print(f"saved {csv_path}")
    print(f"saved {meta_path}")


if __name__ == "__main__":
    main()
