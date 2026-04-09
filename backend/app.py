"""
Thamar's mission planning and monitoring system backend (.py)
Deps: pip install flask flask-cors ultralytics opencv-python numpy Pillow (do it in a venv to avoid conflicts!)
"""

import os, json, heapq, math, base64, io
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "best.pt"

app = Flask(__name__)
CORS(app)          # allow the local HTML file to call us

_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not MODEL_PATH.exists():
        return None
    from ultralytics import YOLO
    _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model


# ============ YOLO detection ============
# this part is still under development

@app.route("/detect", methods=["POST"])
def detect():
    """
    POST  { image: <base64 data-url or raw base64>, conf: 0.25 }
    Returns list of { id, cx, cy, w, h, conf }  (all in image pixels)
    """
    body = request.get_json(force=True)
    if not body or "image" not in body:
        return jsonify(error="Missing 'image' field"), 400

    raw = body["image"]
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]

    img_bytes = base64.b64decode(raw)
    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    conf_thr  = float(body.get("conf", 0.25))

    model = get_model()
    if model is None:
        return jsonify(error=f"Model not found at {MODEL_PATH}. Drop best.pt into backend/models/"), 503

    results = model(np.array(pil_img), conf=conf_thr, verbose=False)[0]

    trees = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        trees.append({
            "id":   i,
            "cx":   round((x1 + x2) / 2),
            "cy":   round((y1 + y2) / 2),
            "w":    round(x2 - x1),
            "h":    round(y2 - y1),
            "conf": round(float(box.conf[0]), 3),
        })

    return jsonify(trees=trees)


# ============ Path Planning ============
# We're using NN's and Dijkstra's algorithms to compute the optimal mission path

def _dist(a, b):
    return math.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"])


def dijkstra_order(nodes):
    """
    Nearest-neighbour seed + Dijkstra-style relaxation over the complete graph.

    For small N (≤ ~15) we do exact TSP via Held-Karp bitmask DP.
    For larger N we fall back to nearest-neighbour greedy (fast, good enough
    for drone routing where trees have spatial locality).
    """
    n = len(nodes)
    if n <= 1:
        return list(range(n))

    if n <= 15:
        return _held_karp(nodes)
    return _nearest_neighbour(nodes)


def _held_karp(nodes):
    n   = len(nodes)
    INF = float("inf")

    dist = [[_dist(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]

    dp   = [[INF] * n for _ in range(1 << n)]
    prev = [[-1]  * n for _ in range(1 << n)]

    dp[1][0] = 0.0

    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask >> u & 1):
                continue
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if mask >> v & 1:
                    continue
                nmask = mask | (1 << v)
                nd    = dp[mask][u] + dist[u][v]
                if nd < dp[nmask][v]:
                    dp[nmask][v]   = nd
                    prev[nmask][v] = u

    full  = (1 << n) - 1
    last  = min(range(n), key=lambda v: dp[full][v])

    path, mask, cur = [], full, last
    while cur != -1:
        path.append(cur)
        nxt  = prev[mask][cur]
        mask = mask ^ (1 << cur)
        cur  = nxt
    path.reverse()
    return path


def _nearest_neighbour(nodes):
    n       = len(nodes)
    visited = [False] * n
    order   = [0]
    visited[0] = True

    for _ in range(n - 1):
        cur  = order[-1]
        best = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: _dist(nodes[cur], nodes[j])
        )
        visited[best] = True
        order.append(best)

    return order


@app.route("/plan", methods=["POST"])
def plan():
    """
    POST { trees: [ {id, cx, cy, ...}, ... ] }
    Returns { order: [idx, ...], waypoints: [{id, cx, cy}, ...] }
    The indices in 'order' refer to positions in the input array.
    """
    body = request.get_json(force=True)
    if not body or "trees" not in body:
        return jsonify(error="Missing 'trees' field"), 400

    trees = body["trees"]
    if len(trees) < 2:
        return jsonify(error="Need at least 2 trees"), 400

    order     = dijkstra_order(trees)
    waypoints = [trees[i] for i in order]

    return jsonify(order=order, waypoints=waypoints)


# ============ Mission Files Generation ============
# Mission details are exported as a .json file 

MISSIONS_DIR = Path(__file__).parent / "missions"
MISSIONS_DIR.mkdir(exist_ok=True)


@app.route("/missions", methods=["GET"])
def list_missions():
    missions = []
    for f in sorted(MISSIONS_DIR.glob("*.json")):
        try:
            meta = json.loads(f.read_text())
            missions.append({
                "id":        f.stem,
                "name":      meta.get("name", f.stem),
                "farm":      meta.get("farm", ""),
                "createdAt": meta.get("createdAt", ""),
                "waypointCount": len(meta.get("waypoints", [])),
            })
        except Exception:
            pass
    return jsonify(missions=missions)


@app.route("/missions/<mid>", methods=["GET"])
def get_mission(mid):
    f = MISSIONS_DIR / f"{mid}.json"
    if not f.exists():
        return jsonify(error="Not found"), 404
    return jsonify(json.loads(f.read_text()))


@app.route("/missions", methods=["POST"])
def save_mission():
    body = request.get_json(force=True)
    if not body or "name" not in body:
        return jsonify(error="Missing 'name'"), 400

    import uuid, datetime
    mid = body.get("id") or uuid.uuid4().hex[:8]
    body["id"]        = mid
    body["createdAt"] = body.get("createdAt") or datetime.datetime.utcnow().isoformat()

    (MISSIONS_DIR / f"{mid}.json").write_text(json.dumps(body, indent=2))
    return jsonify(id=mid, ok=True)


@app.route("/missions/<mid>", methods=["DELETE"])
def delete_mission(mid):
    f = MISSIONS_DIR / f"{mid}.json"
    if not f.exists():
        return jsonify(error="Not found"), 404
    f.unlink()
    return jsonify(ok=True)


# ============ Health check ============

@app.route("/health")
def health():
    return jsonify(
        status="ok",
        model_loaded=_yolo_model is not None,
        model_path=str(MODEL_PATH),
        model_exists=MODEL_PATH.exists(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)