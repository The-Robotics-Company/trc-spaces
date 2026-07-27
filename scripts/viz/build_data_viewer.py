"""Build a self-contained, locally-openable web viewer for mlspaces datagen output.

Modeled on piper-x-policy's cubes data viewer (gallery + per-episode scrubber),
adapted to the mlspaces trajectory format: ``trajectories*.h5`` files (one
``traj_N`` group per episode, see molmo_spaces/utils/save_utils.py) with
sidecar per-camera MP4s in the same directory.

Generates a static site:

  * index.html — dataset stats (episodes, success rate, total steps), filters
    (all / success / fail), sort, one card per episode with an RGB thumbnail,
    HOVER-TO-PLAY video preview and an OK/FAIL badge.
  * episodes/ep_XXXX.html — every camera's RGB (and depth) video playing in
    sync under ONE scrubber, plus trajectory plots (TCP xyz, gripper state vs
    command, joint positions, per-key actions) with a synced time cursor, and
    episode metadata (task description, source file, result).

Videos are copied into the site (assets/videos/) so the output directory is
fully self-contained: open index.html straight from file://, or serve it
(python -m http.server) — no external CDN, no backend.

    python scripts/viz/build_data_viewer.py <datagen_output_dir> [--out DIR]
    # e.g.
    python scripts/viz/build_data_viewer.py \
        experiment_output/datagen/piper_x_cubes_in_cup_v1/PiperXCubesInCupDataGenConfig/20260724_154509
    # then just open results/data_viewer/index.html, or:
    python -m http.server -d results/data_viewer 8009
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None

try:
    from PIL import Image
except ImportError:
    Image = None

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# dataset reading
# --------------------------------------------------------------------------


def decode_json_rows(ds) -> list:
    """Datasets saved as (T, str_max_len) uint8 hold one null-padded JSON doc per row."""
    raw = np.asarray(ds, dtype=np.uint8)
    out = []
    for row in raw:
        try:
            out.append(json.loads(bytes(row).split(b"\x00", 1)[0].decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            out.append({})
    return out


def json_ts_to_array(docs: list, keys=("arm", "gripper")) -> np.ndarray | None:
    """Stack per-step dicts of lists ({'arm': [...], 'gripper': [...]}) into (T, D)."""
    rows = []
    for d in docs:
        if not isinstance(d, dict):
            return None
        vec = []
        for k in keys:
            v = d.get(k, [])
            if isinstance(v, (list, tuple)):
                vec.extend(float(x) for x in v)
        rows.append(vec)
    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return None
    if any(len(r) != width for r in rows):
        rows = [r + [np.nan] * (width - len(r)) for r in rows]
    return np.asarray(rows, dtype=np.float32)


def natural_traj_order(keys):
    def k(name):
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return 1 << 30

    return sorted(keys, key=k)


def parse_obs_scene(ds) -> dict:
    raw = ds[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    elif not isinstance(raw, str):
        raw = str(raw)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def video_meta(path: Path):
    """(fps, n_frames, middle-frame ndarray or None)."""
    if VideoReader is None:
        return None, 0, None
    try:
        vr = VideoReader(str(path))
        n = len(vr)
        fps = float(vr.get_avg_fps())
        mid = vr[n // 2].asnumpy() if n else None
        return fps, n, mid
    except Exception as e:  # unreadable video: card still renders, no thumb
        print(f"[viewer]   WARNING: cannot read {path.name}: {e}")
        return None, 0, None


def collect_episodes(data_root: Path):
    """Walk data_root for trajectories*.h5; yield one record per traj group."""
    h5_paths = sorted(data_root.rglob("trajectories*.h5"))
    if not h5_paths:
        sys.exit(f"[viewer] no trajectories*.h5 found under {data_root}")
    episodes = []
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            for key in natural_traj_order(f.keys()):
                if not key.startswith("traj_"):
                    continue
                g = f[key]
                traj_idx = int(key.split("_")[-1])
                rec = {"h5": h5_path, "traj": key}
                # per-timestep success flags; the episode verdict is the final one
                succ = np.asarray(g["success"]) if "success" in g else np.zeros(1, bool)
                rec["success"] = bool(succ[-1])
                # ---- timeseries -------------------------------------------
                extra = g.get("obs/extra", {})
                rec["tcp"] = (
                    np.asarray(extra["tcp_pose"], np.float32) if "tcp_pose" in extra else None
                )
                # qpos / actions are stored as per-step JSON docs of {'arm','gripper',...}
                qpos = None
                if "obs/agent/qpos" in g:
                    qpos = json_ts_to_array(decode_json_rows(g["obs/agent/qpos"]))
                rec["qpos"] = qpos
                rec["actions"] = {}
                if "actions" in g:
                    for ak in g["actions"]:
                        ds = g["actions"][ak]
                        arr = None
                        if ds.dtype == np.uint8 and ds.ndim == 2:  # JSON rows
                            arr = json_ts_to_array(decode_json_rows(ds))
                        elif ds.dtype.kind == "f" and ds.ndim <= 2:
                            arr = np.asarray(ds, np.float32)
                            if arr.ndim == 1:
                                arr = arr[:, None]
                        if arr is not None and arr.size and arr.shape[1] <= 16:
                            rec["actions"][ak] = arr
                rec["steps"] = int(
                    next(
                        (
                            len(v)
                            for v in [rec["qpos"], rec["tcp"], *rec["actions"].values()]
                            if v is not None
                        ),
                        len(succ),
                    )
                )
                rec["reward_sum"] = (
                    float(np.asarray(g["rewards"]).sum()) if "rewards" in g else 0.0
                )
                scene = parse_obs_scene(g["obs_scene"]) if "obs_scene" in g else {}
                rec["task_description"] = (
                    scene.get("task_description") or scene.get("text") or ""
                )
                rec["object_name"] = scene.get("object_name") or ""
                # ---- videos ------------------------------------------------
                # sensor_data is empty in current datagen output (camera frames are
                # dropped before batching), so discover the sidecar MP4s by the
                # writer's naming convention: episode_{idx:08d}_{camera}{suffix}.mp4
                # with suffix taken from the h5 filename (trajectories{suffix}.h5).
                suffix = h5_path.stem[len("trajectories"):]
                rec["videos"] = {}
                prefix = f"episode_{traj_idx:08d}_"
                for mp4 in sorted(h5_path.parent.glob(f"{prefix}*{suffix}.mp4")):
                    cam = mp4.name[len(prefix):]
                    if suffix and cam.endswith(suffix + ".mp4"):
                        cam = cam[: -len(suffix + ".mp4")]
                    else:
                        cam = cam[: -len(".mp4")]
                    rec["videos"][cam] = mp4
                episodes.append(rec)
    return episodes


# --------------------------------------------------------------------------
# site generation
# --------------------------------------------------------------------------


def b64f(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode()


def jpeg_data_uri(img: np.ndarray, max_w=240, quality=78) -> str:
    if Image is None or img is None:
        return ""
    im = Image.fromarray(img)
    im.thumbnail((max_w, max_w))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(description="Build the mlspaces datagen data viewer.")
    ap.add_argument("data_root", help="Datagen output dir (searched recursively for h5s).")
    ap.add_argument("--out", default=str(REPO / "results" / "data_viewer"))
    ap.add_argument("--title", default=None, help="Site title (default: data_root name).")
    ap.add_argument(
        "--demos",
        default=str(REPO / "results" / "sampling_demos"),
        help="Dir with manifest.json from scripts/viz/record_sampling_demos.py "
        "(adds the 'Sampling demos' section; silently skipped if missing).",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    episodes = collect_episodes(data_root)
    print(f"[viewer] {len(episodes)} episodes from {data_root}")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "episodes").mkdir(parents=True)
    assets = out / "assets"
    (assets / "videos").mkdir(parents=True)
    (assets / "viewer.css").write_text(VIEWER_CSS)
    (assets / "app.js").write_text(APP_JS)
    # three.js for the interactive camera-frustum demo pane (vendored, local)
    vendor = REPO / "scripts" / "viz" / "viewer_vendor"
    for fn in ("three.min.js", "OrbitControls.js"):
        if (vendor / fn).exists():
            shutil.copy(vendor / fn, assets / fn)

    title = args.title or data_root.name
    cards = []
    total_steps = 0
    for ei, rec in enumerate(episodes):
        # ---- copy videos + probe metadata ----------------------------------
        vids = []  # [{cam, file, fps, nframes, depth}]
        thumb = ""
        for cam, src in sorted(rec["videos"].items(), key=lambda kv: kv[0]):
            fps, nf, mid = video_meta(src)
            dst_name = f"ep{ei:04d}_{cam}.mp4"
            shutil.copy(src, assets / "videos" / dst_name)
            vids.append(
                {
                    "cam": cam,
                    "file": f"assets/videos/{dst_name}",
                    "fps": fps or 20.0,
                    "nframes": nf,
                    "depth": cam.endswith("_depth"),
                }
            )
            if not thumb and not cam.endswith("_depth") and mid is not None:
                thumb = jpeg_data_uri(mid)
        # prefer an exo camera as thumb/hover source if present
        rgb_vids = [v for v in vids if not v["depth"]]
        hover = next(
            (v["file"] for v in rgb_vids if "exo" in v["cam"]),
            rgb_vids[0]["file"] if rgb_vids else "",
        )

        # ---- per-episode inline data ----------------------------------------
        series = {}  # name -> {b64, dim, n}
        if rec["tcp"] is not None:
            series["tcp"] = {"data": b64f(rec["tcp"][:, :3]), "dim": 3, "n": len(rec["tcp"])}
        if rec["qpos"] is not None:
            q = rec["qpos"]
            series["qpos"] = {"data": b64f(q), "dim": q.shape[1], "n": len(q)}
        # gripper: measured joint (last qpos cols) vs commanded (last joint_pos col)
        jp = rec["actions"].get("joint_pos")
        if rec["qpos"] is not None and jp is not None and jp.shape[1] > 6:
            n = min(len(rec["qpos"]), len(jp))
            grip = np.stack([rec["qpos"][:n, 6], jp[:n, -1]], axis=1)
            series["gripper"] = {"data": b64f(grip), "dim": 2, "n": n}
        for ak, a in rec["actions"].items():
            series[f"act:{ak}"] = {"data": b64f(a), "dim": a.shape[1], "n": len(a)}

        fps_master = vids[0]["fps"] if vids else 20.0
        ep_data = {
            "ei": ei,
            "n": len(episodes),
            "success": int(rec["success"]),
            "steps": rec["steps"],
            "fps": fps_master,
            "videos": vids,
            "series": series,
            "task": rec["task_description"],
            "object": rec["object_name"],
            "reward": round(rec["reward_sum"], 3),
            "source": f"{rec['h5'].name}::{rec['traj']}",
        }
        html = (
            EP_HTML.replace("__EI__", f"{ei:04d}")
            .replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(ep_data))
            .replace("__PREV__", f"ep_{max(0, ei - 1):04d}.html")
            .replace("__NEXT__", f"ep_{min(len(episodes) - 1, ei + 1):04d}.html")
        )
        (out / "episodes" / f"ep_{ei:04d}.html").write_text(html)

        total_steps += rec["steps"]
        cards.append(
            {
                "ei": ei,
                "success": int(rec["success"]),
                "steps": rec["steps"],
                "thumb": thumb,
                "hover": hover,
                "task": rec["task_description"],
                "ncams": len(rgb_vids),
            }
        )
        print(
            f"[viewer]   episode {ei:3d}: {rec['steps']} steps, "
            f"{len(vids)} videos ({'OK' if rec['success'] else 'FAIL'})"
        )

    n_ok = sum(c["success"] for c in cards)
    stats = {
        "title": title,
        "path": str(data_root),
        "n_eps": len(cards),
        "n_ok": n_ok,
        "rate": round(100.0 * n_ok / max(len(cards), 1), 1),
        "total_steps": total_steps,
    }
    # ---- sampling demos section (optional) ---------------------------------
    demos = []
    demos_dir = Path(args.demos) if args.demos else None
    manifest = demos_dir / "manifest.json" if demos_dir else None
    if manifest and manifest.exists():
        (assets / "demos").mkdir(exist_ok=True)
        for d in json.loads(manifest.read_text()).get("demos", []):
            entry = dict(d)
            media = []
            for m in d.get("media", []):
                src = demos_dir / m["file"]
                if not src.exists():
                    print(f"[viewer]   WARNING: demo media missing: {src}")
                    continue
                if m.get("type") == "cams3d":
                    # camera poses + data-URI images, inlined so the three.js
                    # pane works from file:// (fetch/textures are blocked there)
                    media.append({**m, "data": json.loads(src.read_text())})
                else:
                    shutil.copy(src, assets / "demos" / m["file"])
                    # mtime cache-buster: same filename across re-records, so
                    # browsers would otherwise keep serving the stale image
                    v = int(src.stat().st_mtime)
                    media.append({**m, "file": f"assets/demos/{m['file']}?v={v}"})
            entry["media"] = media
            links = []
            for lk in d.get("links", []):
                src = demos_dir / lk["file"]
                if src.exists():
                    shutil.copy(src, assets / "demos" / lk["file"])
                    links.append({**lk, "file": f"assets/demos/{lk['file']}"})
            entry["links"] = links
            demos.append(entry)
        print(f"[viewer] sampling demos: {len(demos)} entries from {demos_dir}")

    index = (
        INDEX_HTML.replace("__TITLE__", title)
        .replace("__STATS__", json.dumps(stats))
        .replace("__CARDS__", json.dumps(cards))
        .replace("__DEMOS__", json.dumps(demos))
    )
    (out / "index.html").write_text(index)

    print(f"\n[viewer] wrote {out}")
    print(f"[viewer] open {out / 'index.html'} in a browser, or:")
    print(f"[viewer]   python -m http.server -d {out} 8009")


# ============================ front-end assets ==============================

VIEWER_CSS = r"""
:root{
  --bg:#0f1116; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a;
  --fg:#e6e9ef; --muted:#9aa3b2; --accent:#5aa9ff; --ok:#37d67a; --fail:#ff6b6b;
  --chip:#252b36;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}
header.top{padding:18px 24px;border-bottom:1px solid var(--line);
  display:flex;flex-wrap:wrap;gap:16px 28px;align-items:baseline;background:var(--panel)}
header.top h1{font-size:18px;margin:0;font-weight:650}
header.top .path{color:var(--muted);font-size:12px;word-break:break-all}
.stat{display:flex;flex-direction:column;gap:2px}
.stat b{font-size:20px;font-weight:700}
.stat span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.controls{padding:14px 24px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;
  border-bottom:1px solid var(--line)}
.controls .grp{display:flex;gap:6px;align-items:center}
button.f,select{background:var(--chip);color:var(--fg);border:1px solid var(--line);
  border-radius:7px;padding:6px 11px;cursor:pointer;font-size:13px}
button.f.active{background:var(--accent);color:#04121f;border-color:var(--accent);font-weight:650}
.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
  gap:16px;padding:20px 24px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
  overflow:hidden;transition:transform .09s,border-color .09s;display:block}
.card:hover{transform:translateY(-2px);border-color:var(--accent)}
.card .thumbwrap{position:relative;aspect-ratio:16/9;background:#0a0c10;overflow:hidden}
.card .thumbwrap img,.card .thumbwrap video{position:absolute;inset:0;width:100%;height:100%;
  object-fit:cover;display:block}
.card .thumbwrap video{opacity:0;transition:opacity .12s}
.card:hover .thumbwrap video{opacity:1}
.card .play{position:absolute;bottom:6px;right:7px;background:rgba(0,0,0,.55);color:#fff;
  font-size:10px;padding:2px 7px;border-radius:10px;pointer-events:none}
.card:hover .play{opacity:0}
.card .meta{padding:9px 11px;display:flex;justify-content:space-between;align-items:center}
.card .meta .id{font-weight:650}
.badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px}
.badge.ok{background:rgba(55,214,122,.16);color:var(--ok)}
.badge.fail{background:rgba(255,107,107,.16);color:var(--fail)}
.chips{display:flex;gap:6px;padding:0 11px 11px;flex-wrap:wrap}
.chip{font-size:11px;background:var(--chip);color:var(--muted);border-radius:6px;padding:2px 7px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.corner{position:absolute;top:7px;left:7px}

/* sampling demos section */
.demos{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--panel2)}
.demos>h2{margin:0;font-size:13px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);cursor:pointer;user-select:none}
.demos>h2 .tri{display:inline-block;transition:transform .12s;margin-right:6px}
.demos.open>h2 .tri{transform:rotate(90deg)}
.demos .body{display:none;margin-top:14px}
.demos.open .body{display:block}
.democard{background:var(--panel);border:1px solid var(--line);border-radius:11px;
  padding:14px 16px;margin-bottom:14px}
.democard .dhead{display:flex;gap:10px;align-items:center;margin-bottom:4px;flex-wrap:wrap}
.democard .dhead b{font-size:14.5px}
.gchip{font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px;
  background:rgba(90,169,255,.16);color:var(--accent);white-space:nowrap}
.gchip.off{background:var(--chip);color:var(--muted)}
.democard .ddesc{color:var(--muted);font-size:13px;max-width:900px}
.democard .dmedia{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
  gap:12px;margin:12px 0}
.democard .dmedia img,.democard .dmedia video{width:100%;border-radius:8px;
  background:#07090d;display:block}
.democard .medlabel{font-size:11.5px;color:var(--muted);margin-top:4px;text-align:center}
.democard code{display:block;background:#0a0c10;border:1px solid var(--line);
  border-radius:7px;padding:8px 11px;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;
  color:#c9d4e3;overflow-x:auto;white-space:pre;margin:8px 0 6px}
.democard .dmet{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.democard .dlinks{font-size:12.5px;margin-top:7px}
.democard .dmedia .wide{grid-column:1/-1}
.cams3d{height:420px;border-radius:8px;background:#07090d;overflow:hidden;cursor:grab}
.cams3d:active{cursor:grabbing}

/* episode page */
.epwrap{padding:16px 22px;max-width:1500px;margin:0 auto}
.epnav{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:12px}
.epnav .title{font-size:16px;font-weight:650;text-align:center}
.epgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:1050px){.epgrid{grid-template-columns:1fr}}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:12px}
.pane h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.pane video{width:100%;background:#07090d;border-radius:8px;display:block}
.scrub{padding:14px 22px;position:sticky;bottom:0;background:var(--panel2);
  border-top:1px solid var(--line);display:flex;gap:14px;align-items:center;z-index:5}
.scrub input[type=range]{flex:1;accent-color:var(--accent)}
.scrub button{background:var(--accent);color:#04121f;border:none;border-radius:8px;
  padding:8px 16px;font-weight:700;cursor:pointer;font-size:14px}
.scrub .fc{font-variant-numeric:tabular-nums;color:var(--muted);min-width:130px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:6px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:13px}
.kv b{color:var(--muted);font-weight:500}
canvas.plot{width:100%;height:150px;display:block}
"""

APP_JS = r"""
function d64f(b64){const s=atob(b64);const u=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Float32Array(u.buffer);}
const PALETTE=['#ff6b6b','#5aa9ff','#37d67a','#ffd166','#a78bfa','#f78fb5','#7ee0c7','#ff9f43'];

const D=window.EP_DATA;
const SER={};
for(const k in D.series){const s=D.series[k];SER[k]={dim:s.dim,n:s.n,arr:d64f(s.data)};}

// duration is driven by the longest video; plots map [0,dur] -> [0,n)
const vids=[...document.querySelectorAll('video.sync')];
const master=vids[0]||null;
let dur=0;
let playing=false;

function fmtT(t){return t.toFixed(2)+'s';}
const range=document.getElementById('range');
const fc=document.getElementById('fc');
const playBtn=document.getElementById('play');

function setT(t,fromMaster){
  t=Math.max(0,Math.min(dur||0,t));
  range.value=Math.round(1000*t/(dur||1));
  fc.textContent=`${fmtT(t)} / ${fmtT(dur||0)} · step ${Math.round(t*(D.fps||20))}`;
  for(const v of vids){
    if(fromMaster&&v===master)continue;
    if(Math.abs(v.currentTime-t)>(playing?0.12:0.001))v.currentTime=Math.min(t,v.duration||t);
  }
  drawPlots(t/(dur||1));
}

function ready(){
  dur=Math.max(...vids.map(v=>v.duration||0),0.01);
  setT(0,false);
}
let nready=0;
if(vids.length===0){dur=Math.max(...Object.values(SER).map(s=>s.n),1)/ (D.fps||20);setT(0,false);}
for(const v of vids){
  if(v.readyState>=1)  {if(++nready===vids.length)ready();}
  else v.addEventListener('loadedmetadata',()=>{if(++nready===vids.length)ready();});
}

range.addEventListener('input',()=>{pause();setT(+range.value/1000*(dur||1),false);});
function play(){
  playing=true;playBtn.textContent='❚❚ Pause';
  vids.forEach(v=>{if(v.currentTime<v.duration)v.play();});
  requestAnimationFrame(tick);
}
function pause(){playing=false;playBtn.textContent='▶ Play';vids.forEach(v=>v.pause());}
playBtn.addEventListener('click',()=>playing?pause():play());
function tick(){
  if(!playing)return;
  const t=master?master.currentTime:0;
  setT(t,true);
  if(master&&master.ended){pause();}
  requestAnimationFrame(tick);
}
document.addEventListener('keydown',e=>{
  const stepT=1/(D.fps||20);
  if(e.key==='ArrowRight'){pause();setT((master?master.currentTime:0)+stepT,false);}
  if(e.key==='ArrowLeft'){pause();setT((master?master.currentTime:0)-stepT,false);}
  if(e.key===' '){e.preventDefault();playBtn.click();}
});

// ---------- plots ----------
function plot(canvas,ser,frac){
  const cv=canvas;const dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
  const x=cv.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,W,H);
  const n=ser.n,dim=ser.dim,a=ser.arr;
  let mn=Infinity,mx=-Infinity;
  for(let i=0;i<n*dim;i++){const v=a[i];if(v<mn)mn=v;if(v>mx)mx=v;}
  if(!(mx>mn)){mn-=1;mx+=1;}
  const pad=6;
  const X=i=>pad+i/Math.max(1,n-1)*(W-2*pad);
  const Y=v=>H-pad-(v-mn)/(mx-mn)*(H-2*pad);
  if(mn<0&&mx>0){x.strokeStyle='#2a2f3a';x.lineWidth=1;x.beginPath();
    x.moveTo(pad,Y(0));x.lineTo(W-pad,Y(0));x.stroke();}
  for(let d=0;d<dim;d++){
    x.strokeStyle=PALETTE[d%PALETTE.length];x.lineWidth=1.5;x.beginPath();
    for(let i=0;i<n;i++){const px=X(i),py=Y(a[i*dim+d]);i?x.lineTo(px,py):x.moveTo(px,py);}
    x.stroke();
  }
  const cx=pad+frac*(W-2*pad);
  x.strokeStyle='#e6e9ef';x.lineWidth=1;x.beginPath();x.moveTo(cx,0);x.lineTo(cx,H);x.stroke();
}
function drawPlots(frac){
  document.querySelectorAll('canvas.plot').forEach(cv=>{
    const s=SER[cv.dataset.series];if(s)plot(cv,s,frac);
  });
}
window.addEventListener('resize',()=>drawPlots(+range.value/1000));
window.addEventListener('load',()=>drawPlots(0));
"""

EP_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode __EI__ · __TITLE__</title>
<link rel="stylesheet" href="../assets/viewer.css">
</head><body>
<div class="epwrap">
  <div class="epnav">
    <div><a href="../index.html">← all episodes</a></div>
    <div class="title" id="eptitle"></div>
    <div class="grp"><a href="__PREV__">‹ prev</a> &nbsp; <a href="__NEXT__">next ›</a></div>
  </div>
  <div class="epgrid" id="epgrid"></div>
</div>
<div class="scrub">
  <button id="play">▶ Play</button>
  <input type="range" id="range" min="0" max="1000" value="0" step="1">
  <div class="fc" id="fc"></div>
</div>
<script>window.EP_DATA=__DATA__;</script>
<script>
const d=window.EP_DATA;
document.getElementById('eptitle').innerHTML=
  `Episode ${d.ei} / ${d.n-1} `+
  `<span class="badge ${d.success?'ok':'fail'}">${d.success?'SUCCESS':'FAIL'}</span>`;
const grid=document.getElementById('epgrid');
let html='';
for(const v of d.videos){
  html+=`<div class="pane"><h3>${v.cam}${v.depth?' (depth)':''}</h3>`
    +`<video class="sync" src="../${v.file}" muted preload="auto" playsinline></video></div>`;
}
const NICE={tcp:'TCP position (x,y,z)',qpos:'joint positions (qpos: arm + gripper)',
  gripper:'gripper — measured vs commanded'};
for(const k in d.series){
  const label=NICE[k]||('actions — '+k.replace('act:',''));
  html+=`<div class="pane"><h3>${label}</h3><canvas class="plot" data-series="${k}"></canvas></div>`;
}
html+=`<div class="pane"><h3>Episode</h3><div class="kv">`
  +`<b>result</b><span>${d.success?'success':'fail'}</span>`
  +(d.task?`<b>task</b><span>${d.task}</span>`:'')
  +(d.object?`<b>object</b><span>${d.object}</span>`:'')
  +`<b>steps</b><span>${d.steps}</span>`
  +`<b>reward Σ</b><span>${d.reward}</span>`
  +`<b>fps</b><span>${d.fps}</span>`
  +`<b>source</b><span>${d.source}</span>`
  +`</div></div>`;
grid.innerHTML=html;
</script>
<script src="../assets/app.js"></script>
</body></html>"""

INDEX_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · data viewer</title>
<link rel="stylesheet" href="assets/viewer.css">
<script src="assets/three.min.js"></script>
<script src="assets/OrbitControls.js"></script>
</head><body>
<header class="top">
  <div><h1>__TITLE__ — data viewer</h1><div class="path" id="path"></div></div>
  <div class="stat"><b id="s_eps"></b><span>episodes</span></div>
  <div class="stat"><b id="s_rate"></b><span>success rate</span></div>
  <div class="stat"><b id="s_ok"></b><span>successful</span></div>
  <div class="stat"><b id="s_steps"></b><span>total steps</span></div>
</header>
<div class="controls">
  <div class="grp"><span class="muted">result:</span>
    <button class="f active" data-f="res" data-v="all">all</button>
    <button class="f" data-f="res" data-v="ok">success</button>
    <button class="f" data-f="res" data-v="fail">fail</button></div>
  <div class="grp"><span class="muted">sort:</span>
    <select id="sort"><option value="ei">episode #</option>
      <option value="steps">length</option></select></div>
  <div class="grp muted" id="count"></div>
</div>
<div class="demos" id="demos" style="display:none">
  <h2 id="demoshead"><span class="tri">▶</span>Sampling &amp; DR demos <span class="muted" id="democount"></span></h2>
  <div class="body" id="demosbody"></div>
</div>
<div class="grid" id="grid"></div>
<script>
const STATS=__STATS__, CARDS=__CARDS__, DEMOS=__DEMOS__;
document.getElementById('path').textContent=STATS.path;
document.getElementById('s_eps').textContent=STATS.n_eps;
document.getElementById('s_rate').textContent=STATS.rate+'%';
document.getElementById('s_ok').textContent=STATS.n_ok;
document.getElementById('s_steps').textContent=STATS.total_steps.toLocaleString();

// persisted gallery state (filters/sort/scroll survive visiting an episode)
const STORE='mlspacesViewer.state.v1';
let filt={res:'all'};let sort='ei';
function readState(){try{return JSON.parse(sessionStorage.getItem(STORE)||'null');}catch(e){return null;}}
(function(){const s=readState();if(s){if(s.filt)filt=Object.assign(filt,s.filt);if(s.sort)sort=s.sort;}})();
function saveSel(){try{const c=readState()||{};
  sessionStorage.setItem(STORE,JSON.stringify({filt,sort,scrollY:c.scrollY||0}));}catch(e){}}
function saveScroll(){try{sessionStorage.setItem(STORE,JSON.stringify({filt,sort,scrollY:window.scrollY}));}catch(e){}}
window.addEventListener('pagehide',saveSel);
let _raf=0;
window.addEventListener('scroll',()=>{if(!_raf)_raf=requestAnimationFrame(()=>{_raf=0;saveScroll();});},{passive:true});

document.querySelectorAll('button.f').forEach(b=>b.onclick=()=>{
  document.querySelectorAll(`button.f[data-f="${b.dataset.f}"]`).forEach(x=>x.classList.remove('active'));
  b.classList.add('active');filt[b.dataset.f]=b.dataset.v;saveSel();render();});
document.getElementById('sort').onchange=e=>{sort=e.target.value;saveSel();render();};
function syncControls(){
  document.querySelectorAll('button.f').forEach(b=>
    b.classList.toggle('active',String(filt[b.dataset.f])===String(b.dataset.v)));
  document.getElementById('sort').value=sort;
}

function attachHover(cardEl,c){
  if(!c.hover)return;
  const v=cardEl.querySelector('video');if(!v)return;
  cardEl.addEventListener('mouseenter',()=>{
    if(!v.src)v.src=c.hover;      // lazy: only load on first hover
    v.currentTime=0;v.play().catch(()=>{});
  });
  cardEl.addEventListener('mouseleave',()=>{v.pause();});
}

function render(){
  let rows=CARDS.filter(c=>filt.res==='all'||(filt.res==='ok')===(c.success===1));
  rows.sort((a,b)=>sort==='ei'?a.ei-b.ei:b[sort]-a[sort]);
  document.getElementById('count').textContent=`${rows.length} shown`;
  const g=document.getElementById('grid');g.innerHTML='';
  rows.forEach(c=>{
    const a=document.createElement('a');a.className='card';
    a.href=`episodes/ep_${String(c.ei).padStart(4,'0')}.html`;
    a.innerHTML=
      `<div class="thumbwrap">`+
        `<span class="badge ${c.success?'ok':'fail'} corner">${c.success?'OK':'FAIL'}</span>`+
        (c.thumb?`<img src="${c.thumb}" loading="lazy">`
                :`<div style="display:flex;height:100%;align-items:center;justify-content:center" class="muted">no rgb</div>`)+
        (c.hover?`<video muted loop playsinline preload="none"></video><span class="play">▶ hover</span>`:``)+
      `</div>`+
      `<div class="meta"><span class="id">episode ${c.ei}</span>`+
        `<span class="muted">${c.steps} steps</span></div>`+
      `<div class="chips">`+
        (c.task?`<span class="chip" title="${c.task}">${c.task}</span>`:``)+
        `<span class="chip">${c.ncams} cam${c.ncams===1?'':'s'}</span></div>`;
    g.appendChild(a);
    attachHover(a,c);
  });
}
// ---- sampling & DR demos (from record_sampling_demos.py manifest) ----
// interactive pane for 'cams3d' media: each sampled camera as a frustum with
// its rendered view on the image plane (poses are cam-to-world, RDF axes) —
// the static-site equivalent of scripts/viz/piper_x_random_cam_rerun.py
function initCams3D(host,D){
  if(typeof THREE==='undefined'){host.textContent='three.js missing';return;}
  const W=host.clientWidth,H=420;
  const renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio||1);
  renderer.setSize(W,H);renderer.setClearColor(0x07090d,1);
  host.appendChild(renderer.domElement);
  const scene=new THREE.Scene();
  const cam=new THREE.PerspectiveCamera(50,W/H,0.01,30);
  cam.position.set(1.9,-1.3,1.2);cam.up.set(0,0,1);
  const controls=new THREE.OrbitControls(cam,renderer.domElement);
  controls.target.set(D.center[0],D.center[1],D.center[2]);controls.update();
  const grid=new THREE.GridHelper(2.4,24,0x2a2f3a,0x1c2028);
  grid.rotation.x=Math.PI/2;grid.position.set(0.3,0,0);scene.add(grid);
  const aim=new THREE.Mesh(new THREE.SphereGeometry(0.02,16,16),
    new THREE.MeshBasicMaterial({color:0xff4040}));
  aim.position.set(D.center[0],D.center[1],D.center[2]);scene.add(aim);
  const base=new THREE.Mesh(new THREE.SphereGeometry(0.014,16,16),
    new THREE.MeshBasicMaterial({color:0xffffff}));scene.add(base);
  const loader=new THREE.TextureLoader();
  const ar=D.w/D.h;
  for(const s of D.cams){
    const g=new THREE.Group();
    const m=new THREE.Matrix4();m.set(...s.pose);   // row-major 4x4
    g.applyMatrix4(m);
    const Zd=0.25, hh=Zd*Math.tan(s.fov*Math.PI/360), hw=hh*ar;
    const cn=[[-hw,-hh,Zd],[hw,-hh,Zd],[hw,hh,Zd],[-hw,hh,Zd]];
    const V=[];
    for(const c of cn){V.push(0,0,0,c[0],c[1],c[2]);}
    for(let i=0;i<4;i++){const a=cn[i],b=cn[(i+1)%4];V.push(a[0],a[1],a[2],b[0],b[1],b[2]);}
    const fg=new THREE.BufferGeometry();
    fg.setAttribute('position',new THREE.BufferAttribute(new Float32Array(V),3));
    g.add(new THREE.LineSegments(fg,new THREE.LineBasicMaterial({color:0x27c7e6})));
    const pl=new THREE.Mesh(new THREE.PlaneGeometry(2*hw,2*hh),
      new THREE.MeshBasicMaterial({map:loader.load(s.img),side:THREE.DoubleSide}));
    pl.rotation.x=Math.PI;   // RDF: +y is down -> image top at -y
    pl.position.set(0,0,Zd);
    g.add(pl);
    scene.add(g);
  }
  (function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,cam);})();
}
function renderDemos(){
  if(!DEMOS||!DEMOS.length)return;
  const sec=document.getElementById('demos');sec.style.display='';
  document.getElementById('democount').textContent=`(${DEMOS.length})`;
  const host=document.getElementById('demosbody');
  const panes=[];   // [element-id, data] deferred three.js inits
  host.innerHTML=DEMOS.map((d,di)=>{
    const off=(d.group||'').includes('off');
    let media='';
    if(d.media&&d.media.length){
      media='<div class="dmedia">'+d.media.map((m,mi)=>{
        const lab=m.label?`<div class="medlabel">${m.label}</div>`:'';
        if(m.type==='video')
          return `<div><video src="${m.file}" controls muted loop preload="metadata"></video>${lab}</div>`;
        if(m.type==='cams3d'){
          const id=`cams3d_${di}_${mi}`;panes.push([id,m.data]);
          return `<div class="wide"><div class="cams3d" id="${id}"></div>${lab}</div>`;
        }
        return `<div><a href="${m.file}" target="_blank"><img src="${m.file}" loading="lazy"></a>${lab}</div>`;
      }).join('')+'</div>';
    }
    const links=(d.links&&d.links.length)
      ?`<div class="dlinks">${d.links.map(l=>`<a href="${l.file}" download>⬇ ${l.label}</a>`).join(' · ')}</div>`:'';
    return `<div class="democard">`
      +`<div class="dhead"><span class="gchip ${off?'off':''}">${d.group||''}</span><b>${d.title||d.id}</b></div>`
      +`<div class="ddesc">${d.desc||''}</div>`
      +media
      +(d.command?`<code>${d.command}</code>`:'')
      +(d.metrics?`<div class="dmet">${d.metrics}</div>`:'')
      +links
      +`</div>`;
  }).join('');
  const head=document.getElementById('demoshead');
  const KEY='mlspacesViewer.demosOpen';
  // three.js panes must init while visible (display:none => zero clientWidth),
  // so init lazily on the first open
  let inited=false;
  const initPanes=()=>{if(inited)return;inited=true;
    panes.forEach(([id,data])=>initCams3D(document.getElementById(id),data));};
  const setOpen=o=>{sec.classList.toggle('open',o);
    if(o)requestAnimationFrame(initPanes);
    try{sessionStorage.setItem(KEY,o?'1':'0');}catch(e){}};
  head.onclick=()=>setOpen(!sec.classList.contains('open'));
  let open=true;try{open=sessionStorage.getItem(KEY)!=='0';}catch(e){}
  setOpen(open);
}
renderDemos();
syncControls();render();
function restoreScroll(){const s=readState();const y=s&&s.scrollY||0;
  if(y&&Math.abs(window.scrollY-y)>2)window.scrollTo(0,y);}
window.addEventListener('pageshow',()=>{[0,60,200,500].forEach(ms=>setTimeout(restoreScroll,ms));});
</script>
</body></html>"""


if __name__ == "__main__":
    main()
