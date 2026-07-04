"""PitchIQ dashboard — Streamlit frontend.

Five tabs per processed match: annotated video, video-synced tactical radar,
analytics charts, intelligence (roles / similarity / marking), and the
analyst report with grounded Q&A. Reads artifacts straight from the store on
disk; uploads are processed in-process on a background thread so the single
container works standalone (the FastAPI service exposes the same pipeline for
programmatic use / a future React frontend).

Run:  streamlit run pitchiq/app/ui.py
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pitchiq.config import load_config
from pitchiq.core.artifacts import ArtifactStore
from pitchiq.core.env import load_env

load_env()
st.set_page_config(page_title="PitchIQ", page_icon="⚽", layout="wide")

CFG = load_config()
JOBS_ROOT = Path(CFG.app.artifacts_root)
DEMO_ROOT = Path(CFG.app.demo_root)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- helpers
def list_matches() -> list[dict]:
    out = []
    for root, kind in ((DEMO_ROOT, "demo"), (JOBS_ROOT, "upload")):
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            store = ArtifactStore(d)
            name = d.name
            if store.meta_path.exists():
                m = store.load_meta()
                name = f"{m.team_names.get('home', 'Home')} vs {m.team_names.get('away', 'Away')}"
            out.append({"id": d.name, "dir": d, "name": f"{name} [{kind}]",
                        "status": store.read_status()})
    return out


@st.cache_resource(show_spinner=False)
def load_store_bundle(job_dir: str):
    """Heavy artifacts cached per job dir."""
    store = ArtifactStore(job_dir)
    b = {"store": store, "meta": store.load_meta(),
         "tracking": store.load_tracking(),
         "summary": store.load_json(store.analytics_path("summary.json"))}
    for name, path in [
        ("kin", store.analytics_path("kinematics.parquet")),
        ("possession", store.analytics_path("possession.parquet")),
        ("shape", store.analytics_path("shape.parquet")),
        ("marking_tl", store.intelligence_path("marking_timeline.parquet")),
    ]:
        b[name] = pd.read_parquet(path) if path.exists() else None
    for name, path in [
        ("roles", store.intelligence_path("roles.json")),
        ("similarity", store.intelligence_path("similarity.json")),
        ("marking", store.intelligence_path("marking.json")),
        ("facts", store.report_path("facts.json")),
    ]:
        b[name] = store.load_json(path) if path.exists() else None
    hm_path = store.analytics_path("heatmaps.npz")
    b["heatmaps"] = dict(np.load(hm_path)) if hm_path.exists() else {}
    pc_path = store.analytics_path("pitch_control.npz")
    b["pitch_control"] = dict(np.load(pc_path)) if pc_path.exists() else {}
    return b


def player_labels(summary: dict, team_names: dict) -> dict[int, str]:
    out = {}
    for eid, p in summary.get("players", {}).items():
        j = p.get("jersey_no")
        core = f"#{int(j)}" if j is not None else f"id{eid}"
        out[int(eid)] = f"{core} ({team_names.get(p.get('team', ''), p.get('team', '?'))})"
    return out


def start_processing(video_bytes: bytes, filename: str) -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    store = ArtifactStore(JOBS_ROOT / job_id)
    store.input_video.write_bytes(video_bytes)
    store.update_status("queued", 0.0, "queued", state="queued")

    def run() -> None:
        from pitchiq.pipeline.full import FullPipeline

        try:
            FullPipeline(load_config()).process_video(store.input_video, store)
        except Exception:
            pass  # status.json carries the error

    threading.Thread(target=run, daemon=True).start()
    return job_id


# ----------------------------------------------------------------- sidebar
st.sidebar.title("⚽ PitchIQ")
st.sidebar.caption("Tactical intelligence from broadcast video")

matches = list_matches()
ready = [m for m in matches if m["status"].get("state") in ("done", "unknown")
         and ArtifactStore(m["dir"]).has_tracking()]
processing = [m for m in matches if m["status"].get("state") in ("queued", "running")]

if processing:
    st.sidebar.subheader("Processing")
    for m in processing:
        s = m["status"]
        st.sidebar.progress(min(1.0, float(s.get("progress", 0))),
                            text=f"{m['id'][:18]} — {s.get('stage')}: {s.get('message', '')[:38]}")
    if st.sidebar.button("🔄 Refresh"):
        st.rerun()

choice = st.sidebar.selectbox(
    "Match", options=[m["id"] for m in ready],
    format_func=lambda i: next(m["name"] for m in ready if m["id"] == i),
) if ready else None

st.sidebar.divider()
up = st.sidebar.file_uploader("Analyse a new clip", type=["mp4", "mov", "mkv"])
if up is not None and st.sidebar.button("🚀 Process clip"):
    jid = start_processing(up.getvalue(), up.name)
    st.sidebar.success(f"Job {jid} started — this runs the full CV pipeline "
                       "and can take several minutes per minute of video on CPU.")
    time.sleep(1)
    st.rerun()

if not ready:
    st.info("No processed matches yet. Upload a clip, or run "
            "`python scripts/build_demo.py` to generate the bundled demo match.")
    st.stop()

bundle = load_store_bundle(str(next(m["dir"] for m in ready if m["id"] == choice)))
store: ArtifactStore = bundle["store"]
meta = bundle["meta"]
summary = bundle["summary"]
kit = meta.kit_colors
labels = player_labels(summary, meta.team_names)

st.title(f"{meta.team_names.get('home')} vs {meta.team_names.get('away')}")
c1, c2, c3, c4 = st.columns(4)
share = summary.get("possession", {}).get("share", {})
c1.metric("Possession (home)", f"{100 * share.get('home', 0):.0f}%")
tilt = summary.get("field_tilt", {}).get("tilt_home")
c2.metric("Field tilt (home)", f"{tilt:.2f}" if tilt is not None else "—")
c3.metric("Passes detected", summary.get("events", {}).get("n_passes", 0))
ppda = summary.get("ppda", {})
c4.metric("PPDA h / a", f"{ppda.get('home', {}).get('ppda', '—')} / "
                        f"{ppda.get('away', {}).get('ppda', '—')}")

tab_video, tab_radar, tab_analytics, tab_intel, tab_report = st.tabs(
    ["🎬 Annotated video", "🗺️ Tactical map", "📊 Analytics", "🧠 Intelligence",
     "📝 Report"])

# ------------------------------------------------------------------- video
with tab_video:
    ann = store.media_path("annotated.mp4")
    src = store.input_video
    prev = store.media_path("preview.mp4")
    if ann.exists():
        st.video(str(ann))
        st.caption("Boxes, persistent IDs, team colours and jersey numbers are "
                   "drawn from the cached tracking table — re-renderable without "
                   "re-running detection.")
    elif src.exists():
        st.video(str(src))
        st.caption("Annotated render not available — showing source clip.")
    elif prev.exists():
        st.video(str(prev))
        st.caption("Showing the bundled annotated preview (full-resolution "
                   "renders are regenerated by `python scripts/build_demo.py`).")
    else:
        st.info("This match has no source video (data-only import). "
                "The tactical map still works standalone.")

# ------------------------------------------------------------------- radar
with tab_radar:
    from pitchiq.viz.radar_html import build_radar_html

    preview = store.media_path("preview.mp4")
    video_for_radar = preview if preview.exists() else None
    with st.spinner("Building radar…"):
        html = build_radar_html(bundle["tracking"], meta, video_for_radar,
                                radar_fps=CFG.app.radar_fps)
    components.html(html, height=980 if video_for_radar else 640, scrolling=False)
    st.caption("The radar is drawn from projected pitch coordinates and stays "
               "frame-synced to the video via its playback clock — scrub away.")

# --------------------------------------------------------------- analytics
with tab_analytics:
    from pitchiq.viz import charts

    colA, colB = st.columns(2)
    with colA:
        if bundle["possession"] is not None:
            st.plotly_chart(charts.possession_flow_fig(
                bundle["possession"], kit, meta.fps), use_container_width=True)
        team_hm = st.selectbox("Heatmap", [k for k in bundle["heatmaps"]
                                           if k.startswith("team_")] +
                               sorted([k for k in bundle["heatmaps"]
                                       if k.startswith("player_")],
                                      key=lambda k: int(k.split("_")[1])),
                               format_func=lambda k: (
                                   meta.team_names.get(k[5:], k) if k.startswith("team_")
                                   else labels.get(int(k.split("_")[1]), k)))
        if team_hm:
            st.plotly_chart(charts.heatmap_fig(bundle["heatmaps"][team_hm],
                                               f"Occupancy — {team_hm}"),
                            use_container_width=True)
        if bundle["shape"] is not None and len(bundle["shape"]):
            st.plotly_chart(charts.shape_timeline_fig(bundle["shape"], kit, meta.fps),
                            use_container_width=True)
    with colB:
        if "mean_home_control" in bundle["pitch_control"]:
            st.plotly_chart(charts.pitch_control_fig(
                bundle["pitch_control"]["mean_home_control"], kit),
                use_container_width=True)
        pn_team = st.radio("Pass network", ["home", "away"], horizontal=True)
        net = summary.get("pass_network", {}).get(pn_team, {})
        st.plotly_chart(charts.pass_network_fig(
            net, kit.get(pn_team, "#888"), meta.team_names.get(pn_team, pn_team)),
            use_container_width=True)
        st.plotly_chart(charts.phase_share_fig(summary.get("phases", {}),
                                               meta.team_names),
                        use_container_width=True)

    st.subheader("Formations & shape")
    fcols = st.columns(2)
    for i, team in enumerate(("home", "away")):
        f = summary.get("formations", {}).get(team, {})
        s = summary.get("shape", {}).get(team, {})
        with fcols[i]:
            st.markdown(f"**{meta.team_names.get(team)}** — "
                        f"{f.get('shape_morph', 'shape data unavailable')}  \n"
                        f"Block: *{s.get('block_label', '?')}*")
    xt_path = store.analytics_path("xt_players.parquet")
    if xt_path.exists():
        st.plotly_chart(charts.xt_bar_fig(pd.read_parquet(xt_path), labels, kit),
                        use_container_width=True)

# ------------------------------------------------------------- intelligence
with tab_intel:
    roles = bundle["roles"] or {}
    colR, colS = st.columns(2)
    with colR:
        st.subheader("Discovered roles")
        st.caption(f"Unsupervised clustering of style embeddings "
                   f"({roles.get('embedding_method', '?')} embedding, "
                   f"k={roles.get('k', '?')})")
        rows = [{"player": labels.get(int(e), e), "role": info["role"]}
                for e, info in roles.get("players", {}).items()]
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("role"),
                         use_container_width=True, height=330)
        mm = roles.get("nominal_vs_actual", [])
        if mm:
            st.markdown("**Position ≠ role** (nominal slot vs behaviour):")
            for m in mm[:6]:
                st.markdown(f"- {labels.get(m['entity_id'], m['entity_id'])}: "
                            f"nominal **{m['nominal_slot']}** → plays as "
                            f"**{m['discovered_role']}**")
    with colS:
        st.subheader("Similar-player search")
        sim = bundle["similarity"] or {}
        neighbors = sim.get("neighbors", {})
        if neighbors:
            q = st.selectbox("Player", sorted(neighbors, key=int),
                             format_func=lambda e: labels.get(int(e), e))
            st.caption(f"index backend: {sim.get('backend')}")
            for hit in neighbors.get(q, [])[:5]:
                drivers = ", ".join(f"{k} {v:+.2f}" for k, v in
                                    sorted(hit["drivers"].items(),
                                           key=lambda kv: -kv[1])[:3])
                st.markdown(f"- **{labels.get(hit['entity_id'], hit['entity_id'])}** — "
                            f"similarity {hit['similarity']:.2f}  \n"
                            f"  <span style='color:gray;font-size:0.85em'>driven by: "
                            f"{drivers}</span>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Marking analysis — who marks whom")
    marking = bundle["marking"] or {}
    mcols = st.columns(2)
    for i, team in enumerate(("home", "away")):
        op = marking.get(team, {}).get("open_play", {})
        with mcols[i]:
            if "team_man_score" in op:
                st.metric(f"{meta.team_names.get(team)} scheme",
                          op.get("scheme", "?"),
                          f"man-score {op['team_man_score']}")
                for p in op.get("pairs", [])[:5]:
                    st.markdown(f"- {labels.get(p['defender_id'], p['defender_id'])} "
                                f"marks {labels.get(p['attacker_id'], p['attacker_id'])} "
                                f"({100 * p['share']:.0f}%)")
            else:
                st.info(f"{meta.team_names.get(team)}: not enough defensive samples")
    if bundle["marking_tl"] is not None and bundle["kin"] is not None:
        from pitchiq.viz.charts import marking_fig

        tl = bundle["marking_tl"]
        def_team = st.radio("Show assignments while defending:", ["home", "away"],
                            horizontal=True, key="mkteam")
        frames = sorted(tl[tl.defending_team == def_team].frame.unique())
        if frames:
            fsel = st.slider("Defensive moment (frame)", int(frames[0]),
                             int(frames[-1]), int(frames[len(frames) // 2]))
            st.plotly_chart(marking_fig(tl, bundle["kin"], fsel, kit, def_team),
                            use_container_width=True)

# ------------------------------------------------------------------ report
with tab_report:
    rp = store.report_path("report.md")
    meta_rp = store.report_path("report_meta.json")
    if rp.exists():
        gen = store.load_json(meta_rp).get("generator", "?") if meta_rp.exists() else "?"
        st.caption(f"generated by: {gen} — every claim traces to facts.json "
                   "(metrics appendix at the bottom)")
        st.markdown(rp.read_text(encoding="utf-8"))
    else:
        st.info("Report not generated yet.")
    st.divider()
    st.subheader("Ask the match")
    if bundle["facts"]:
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []
        for q_, a_ in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(q_)
            with st.chat_message("assistant"):
                st.write(a_)
        question = st.chat_input("e.g. How did the away team press?")
        if question:
            from pitchiq.report.qa import answer_question

            with st.spinner("Answering from computed metrics…"):
                ans = answer_question(question, bundle["facts"], CFG.report)
            st.session_state.qa_history.append(
                (question, f"{ans['answer']}\n\n*engine: {ans['engine']}*"))
            st.rerun()
