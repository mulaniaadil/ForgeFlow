"""Factory Layout Studio.

Run from the project root:  python -m streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from factory_ai import config as C  # noqa: E402
from factory_ai.layout_optimizer import BIG, Problem, cost_path, optimize, violations  # noqa: E402
from factory_ai.pipeline import Scene, detect_scene  # noqa: E402
from factory_ai.process_flow import PRODUCTS, build_flows  # noqa: E402
from factory_ai.visualization import draw_detections, plot_layout  # noqa: E402

APP_NAME = "Factory Layout Studio"
VERSION = "1.0"
CKPT = ROOT / "outputs" / "detector.pt"
SAMPLE_DIR = ROOT / "data" / "images"
MAX_UPLOAD_MB = 15
TYPES = ["Machine", "Dock", "Column"]
EFFORT = {  # label -> (restarts, iterations per restart, refinement steps)
    "Quick": (3, 8000, 60),
    "Balanced": (6, 20000, 150),
    "Thorough": (10, 35000, 300),
}

st.set_page_config(page_title=APP_NAME, page_icon=":material/factory:", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
.hero { background: linear-gradient(120deg, #0f2540 0%, #1b4a78 100%); color: #fff; border-radius: 14px;
        padding: 1.6rem 1.9rem; margin-bottom: 1.25rem; }
.hero h2 { color: #fff; margin: 0 0 .3rem 0; font-size: 1.65rem; letter-spacing: -.02em; }
.hero p { color: #cfe0f2; margin: 0; font-size: .97rem; }
.eyebrow { color: #2563eb; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.hero .eyebrow { color: #93c5fd; }
[data-testid="stMetricValue"] { font-size: 1.75rem; }
.footer { color: #8a96a8; font-size: .75rem; text-align: center; margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================ model & helpers
@st.cache_resource(show_spinner="Loading detection engine...")
def _load_engine(path: str, mtime: float):
    import torch
    from factory_ai.detector import load_model
    meta = torch.load(path, map_location="cpu", weights_only=False)
    return load_model(path), dict(epoch=meta.get("epoch"), score=meta.get("val_mAP50"))


def engine():
    """(model, info) or (None, reason). Never raises: the rest of the app works without detection."""
    if not CKPT.exists():
        return None, "Detection engine not installed (outputs/detector.pt missing)."
    try:
        return _load_engine(str(CKPT), CKPT.stat().st_mtime)
    except Exception as exc:  # corrupt file, missing torch, ...
        return None, f"Detection engine could not be loaded: {exc}"


@st.cache_data(show_spinner=False, max_entries=24)
def _detect(img_bytes: bytes, bar_m: float, thr: float, manual: float | None, engine_key: float):
    model, _ = engine()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    scene = detect_scene(img, model, bar_m, thr, manual)
    overlay = cv2.cvtColor(draw_detections(img, scene.detections, scale=3), cv2.COLOR_BGR2RGB)
    return scene, overlay


def benchmark_scene() -> Scene:
    layout = {"Cutting": (2.0, 12.0, 4.0, 3.0), "Drilling": (9.0, 12.0, 3.0, 3.0),
              "Assembly": (17.0, 12.0, 5.0, 3.0), "Inspection": (20.0, 5.0, 3.0, 2.5),
              "Packaging": (11.0, 5.0, 4.0, 3.0), "Storage": (2.0, 3.0, 4.0, 4.0)}
    fixed = {"Receiving": (0.0, 17.0, 1.5, 3.0), "Dispatch": (28.5, 1.0, 1.5, 3.0)}
    return Scene(30.0, 20.0, layout, fixed, [(14.0, 9.0, 1.5, 1.5)], 0.0)


def ss():
    return st.session_state


def init_state():
    s = ss()
    if "scene" not in s:
        s.scene = benchmark_scene()
        s.locked = set()                       # machines the user pinned in place
        s.source = "Reference plant (30 × 20 m)"
        s.from_detection = False
        s.products = {k: dict(v) for k, v in PRODUCTS.items()}
        s.rev = 0                              # bumps on every model change -> invalidates results
        s.result = None


def commit(scene=None, products=None, locked=None, source=None, from_detection=None):
    s = ss()
    if scene is not None:
        s.scene = scene
    if products is not None:
        s.products = products
    if locked is not None:
        s.locked = set(locked)
    if source is not None:
        s.source = source
    if from_detection is not None:
        s.from_detection = from_detection
    s.rev += 1


def movable_fixed(scene):
    """Split machines into movable layout and fixed objects (docks + locked machines)."""
    locked = ss().locked
    layout = {n: r for n, r in scene.layout.items() if n not in locked}
    fixed = dict(scene.fixed)
    fixed.update({n: r for n, r in scene.layout.items() if n in locked})
    return layout, fixed


def current_flows():
    sc = ss().scene
    return build_flows(ss().products, available=set(sc.layout) | set(sc.fixed))


def build_problem(clearance=1.0, tolerance=0.0):
    sc = ss().scene
    layout, fixed = movable_fixed(sc)
    machines = {n: (r[2], r[3]) for n, r in layout.items()}
    flows = {k: v for k, v in current_flows().items() if v > 0}
    return Problem(sc.floor_w, sc.floor_h, machines, flows, fixed, list(sc.obstacles),
                   clearance=clearance + tolerance, wall_margin=0.5, grid=0.5, allow_rotation=True), layout


def figure(problem, layout, title="", paths=0):
    fig = plot_layout(problem, layout, title, show_flows=True, show_paths=paths)
    fig.patch.set_facecolor("white")
    return fig


def show(fig, container=None):
    (container or st).pyplot(fig, width="stretch")
    plt.close(fig)


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fmt(v, unit=""):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.0f}{unit}"


def header(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def result_is_current():
    r = ss().result
    return r is not None and r["rev"] == ss().rev


# ============================================================================ pages
def page_dashboard():
    st.markdown(f"""<div class="hero"><div class="eyebrow">Layout planning workspace</div>
        <h2>{APP_NAME}</h2>
        <p>Turn a floor plan into an editable digital layout and find the equipment arrangement that cuts
        material-handling distance — in minutes.</p></div>""", unsafe_allow_html=True)

    sc = ss().scene
    problem, layout = build_problem()
    issues = violations(problem, layout) if layout else []
    r = ss().result

    c = st.columns(4)
    c[0].metric("Floor area", f"{sc.floor_w * sc.floor_h:,.0f} m²", f"{sc.floor_w:g} × {sc.floor_h:g} m",
                delta_color="off", delta_arrow="off", border=True)
    c[1].metric("Equipment", len(sc.layout), f"{len(ss().locked)} locked" if ss().locked else "all movable",
                delta_color="off", delta_arrow="off", border=True)
    c[2].metric("Layout check", "Pass" if not issues else f"{len(issues)} issue(s)",
                "clearances OK" if not issues else "review in Layout editor", delta_color="off",
                delta_arrow="off", border=True)
    if result_is_current():
        c[3].metric("Travel distance saved", f"{r['saving_pct']:.1f}%", "latest optimization",
                    delta_color="off", delta_arrow="off", border=True)
    else:
        c[3].metric("Travel distance saved", "—", "not optimized yet", delta_color="off",
                    delta_arrow="off", border=True)

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"**Current layout** · {ss().source}")
            show(figure(problem, sc.layout))
    with right:
        st.subheader("Get started")
        steps = [
            ("pages/import", ":material/upload_file:", "Import a floor plan",
             "Detect machines, docks and columns from a plan image."),
            ("pages/editor", ":material/edit_square:", "Review the layout",
             "Correct positions, sizes and lock equipment in place."),
            ("pages/flow", ":material/swap_horiz:", "Set production volumes",
             "Define how much material moves between stations."),
            ("pages/optimize", ":material/auto_awesome:", "Optimize & export",
             "Generate an improved arrangement and download it."),
        ]
        for i, (key, icon, title, desc) in enumerate(steps, 1):
            with st.container(border=True):
                st.page_link(PAGES[key], label=f"{i}. {title}", icon=icon)
                st.caption(desc)


def page_import():
    header("Step 1", "Import floor plan", "Detect equipment automatically from a top-down plan image.")
    model, info = engine()
    if model is None:
        st.error(info, icon=":material/error:")
        st.info("You can still build a layout manually in the **Layout editor**.", icon=":material/info:")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Plan image**")
            samples = sorted(SAMPLE_DIR.glob("test_*.png"))[:30]
            options = ["Upload image"] + (["Sample plan"] if samples else [])
            mode = st.segmented_control("Source", options, default=options[-1], label_visibility="collapsed")
            img_bytes, name = None, None
            if mode == "Upload image":
                up = st.file_uploader("PNG or JPG, top-down, with outer wall and scale bar",
                                      type=["png", "jpg", "jpeg"])
                if up is not None:
                    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                        st.error(f"File exceeds {MAX_UPLOAD_MB} MB.")
                    else:
                        img_bytes, name = up.getvalue(), up.name
            elif mode == "Sample plan":
                pick = st.selectbox("Sample plan", samples, format_func=lambda p: f"Plant plan {p.stem.split('_')[-1]}")
                img_bytes, name = pick.read_bytes(), pick.name

        with st.container(border=True):
            st.markdown("**Scale & sensitivity**")
            bar_m = st.number_input("Scale bar length (m)", 0.5, 100.0, float(C.SCALE_BAR_M), 0.5)
            manual = st.number_input("Fixed scale (px per m)", 0.0, 1000.0, 0.0, 0.5,
                                     help="Leave at 0 to read the scale bar automatically.")
            thr = st.slider("Detection confidence", 0.10, 0.95, 0.50, 0.05,
                            help="Lower finds more objects but may add false detections.")
        if isinstance(info, dict) and info.get("score") is not None:
            st.caption(f"Detection engine · model build {info.get('epoch')} · validation accuracy "
                       f"(mAP@0.5) {info['score']:.0%}")

    with right:
        if img_bytes is None:
            with st.container(border=True):
                st.info("Upload a plan image or choose a sample plan to start.", icon=":material/image:")
            return
        if cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR) is None:
            st.error("This file is not a readable image.", icon=":material/broken_image:")
            return
        try:
            with st.spinner("Analysing plan..."):
                scene, overlay = _detect(img_bytes, float(bar_m), float(thr), float(manual) or None,
                                         CKPT.stat().st_mtime)
        except Exception as exc:
            st.error(f"**Could not read the plan scale.** {exc}  \nEnter a fixed scale (px per m) and retry.",
                     icon=":material/error:")
            return

        missing = [n for n in C.MACHINES + C.DOCKS if n not in scene.layout and n not in scene.fixed]
        ok = lambda found, total: "green" if found == total else "orange"
        nm, nd = len(scene.layout), len(scene.fixed)
        st.markdown(f":{ok(nm, len(C.MACHINES))}-badge[:material/precision_manufacturing: {nm}/{len(C.MACHINES)} machines] "
                    f":{ok(nd, len(C.DOCKS))}-badge[:material/local_shipping: {nd}/{len(C.DOCKS)} docks] "
                    f":gray-badge[:material/view_column: {len(scene.obstacles)} column{'s' * (len(scene.obstacles) != 1)}] "
                    f":blue-badge[:material/straighten: {scene.floor_w:.1f} × {scene.floor_h:.1f} m · "
                    f"{scene.px_per_m:.1f} px/m]")

        with st.container(border=True):
            st.image(overlay, caption=name, width="stretch")
        if missing:
            st.warning(f"**Not found:** {', '.join(missing)}. Add them in the Layout editor after import, "
                       "or lower the detection confidence.", icon=":material/warning:")
        else:
            st.success("All machines and docks were found.", icon=":material/check_circle:")
        for w in scene.warnings:
            if not w.endswith("not detected"):
                st.info(w.capitalize(), icon=":material/info:")

        if st.button("Use this layout", type="primary", icon=":material/check:", width="stretch",
                     disabled=not (scene.layout or scene.fixed)):
            commit(scene=Scene(scene.floor_w, scene.floor_h, dict(scene.layout), dict(scene.fixed),
                               list(scene.obstacles), scene.px_per_m),
                   locked=set(), source=f"Imported from {name}", from_detection=True)
            st.switch_page(PAGES["pages/editor"])


def _scene_frame(scene):
    rows = [dict(Name=n, Type="Machine", x=r[0], y=r[1], Width=r[2], Depth=r[3], Locked=n in ss().locked)
            for n, r in scene.layout.items()]
    rows += [dict(Name=n, Type="Dock", x=r[0], y=r[1], Width=r[2], Depth=r[3], Locked=True)
             for n, r in scene.fixed.items()]
    rows += [dict(Name=f"Column {i + 1}", Type="Column", x=r[0], y=r[1], Width=r[2], Depth=r[3], Locked=True)
             for i, r in enumerate(scene.obstacles)]
    return pd.DataFrame(rows, columns=["Name", "Type", "x", "y", "Width", "Depth", "Locked"]).astype(
        {"x": float, "y": float, "Width": float, "Depth": float, "Locked": bool})


def _parse_frame(df, floor_w, floor_h):
    """Validate the edited table -> (Scene, locked, errors, notes)."""
    errors, notes = [], []
    df = df.copy()
    df["Name"] = df["Name"].astype("string").str.strip()
    incomplete = df[["Name", "Type", "x", "y", "Width", "Depth"]].isna().any(axis=1) | (df["Name"] == "")
    if incomplete.any():
        notes.append(f"{int(incomplete.sum())} incomplete row(s) were skipped.")
    df = df[~incomplete]
    if ((df.Width <= 0) | (df.Depth <= 0)).any():
        errors.append("Width and depth must be greater than zero.")
    named = df[df.Type != "Column"]
    dup = named.Name[named.Name.duplicated()].unique().tolist()
    if dup:
        errors.append(f"Duplicate names: {', '.join(dup)}. Each machine and dock needs a unique name.")
    layout, fixed, cols, locked = {}, {}, [], set()
    for r in df.itertuples():
        rect = (float(r.x), float(r.y), float(r.Width), float(r.Depth))
        if r.x + r.Width > floor_w + 1e-6 or r.y + r.Depth > floor_h + 1e-6:
            notes.append(f"{r.Name} extends beyond the floor boundary.")
        if r.Type == "Column":
            cols.append(rect)
        elif r.Type == "Dock":
            fixed[r.Name] = rect
        else:
            layout[r.Name] = rect
            if bool(r.Locked):
                locked.add(r.Name)
    unknown = [n for n in list(layout) + list(fixed) if n not in C.MACHINES + C.DOCKS]
    if unknown:
        notes.append(f"No production route uses {', '.join(unknown)} — it is treated as space-taking "
                     "equipment without material flow.")
    return Scene(floor_w, floor_h, layout, fixed, cols, ss().scene.px_per_m), locked, errors, notes


def page_editor():
    header("Step 2", "Layout editor", "Review and correct every object on the floor. All dimensions are in metres; "
           "x, y is the lower-left corner measured from the bottom-left of the floor.")
    sc = ss().scene
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.form(f"editor_form_{ss().rev}", border=True):
            f1, f2 = st.columns(2)
            fw = f1.number_input("Floor width (m)", 5.0, 500.0, float(sc.floor_w), 0.5)
            fh = f2.number_input("Floor depth (m)", 5.0, 500.0, float(sc.floor_h), 0.5)
            num = dict(min_value=0.0, step=0.1, format="%.2f")
            edited = st.data_editor(
                _scene_frame(sc), num_rows="dynamic", hide_index=True, width="stretch",
                column_config={
                    "Name": st.column_config.TextColumn("Name", required=True),
                    "Type": st.column_config.SelectboxColumn("Type", options=TYPES, required=True, default="Machine"),
                    "x": st.column_config.NumberColumn("x", **num),
                    "y": st.column_config.NumberColumn("y", **num),
                    "Width": st.column_config.NumberColumn("Width", **num),
                    "Depth": st.column_config.NumberColumn("Depth", **num),
                    "Locked": st.column_config.CheckboxColumn("Locked", default=False,
                                                              help="Locked machines stay in place during optimization."),
                })
            st.caption("Add rows with ➕ at the bottom of the table; select rows and press Delete to remove them.")
            b1, b2 = st.columns([1, 1])
            save = b1.form_submit_button("Save changes", type="primary", icon=":material/save:", width="stretch")
            reset = b2.form_submit_button("Load reference plant", icon=":material/restart_alt:", width="stretch")

        if reset:
            commit(scene=benchmark_scene(), locked=set(), source="Reference plant (30 × 20 m)", from_detection=False)
            st.rerun()
        if save:
            new, locked, errors, notes = _parse_frame(edited, fw, fh)
            if errors:
                for e in errors:
                    st.error(e, icon=":material/error:")
            else:
                commit(scene=new, locked=locked)
                ss().editor_notes = notes
                st.toast("Layout saved", icon=":material/check:")
                st.rerun()
        for n in ss().pop("editor_notes", []):
            st.warning(n, icon=":material/warning:")

    problem, layout = build_problem()
    issues = violations(problem, layout) if layout else []
    with right:
        with st.container(border=True):
            show(figure(problem, sc.layout, "Saved layout"))
        if not layout:
            st.info("No movable machines — add or unlock at least one machine to optimize.", icon=":material/info:")
        elif issues:
            with st.expander(f":material/report: {len(issues)} clearance issue(s) (1.0 m minimum)"):
                st.markdown("\n".join(f"- {i}" for i in issues))
                st.caption("The optimizer always returns an arrangement that resolves these.")
        else:
            st.success("All objects respect the 1.0 m minimum clearance.", icon=":material/check_circle:")


def page_flow():
    header("Step 3", "Material flow", "Production volumes drive the optimization: stations that exchange more "
           "material are placed closer together.")
    rows = [dict(Product=k, Volume=int(p["volume"]), Route=" → ".join(p["route"])) for k, p in ss().products.items()]
    with st.form(f"flow_form_{ss().rev}", border=True):
        st.markdown("**Product families** · unit loads moved per shift")
        ed = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["Product", "Route"],
                            column_config={
                                "Volume": st.column_config.NumberColumn("Loads / shift", min_value=0, max_value=10000, step=10),
                                "Route": st.column_config.TextColumn("Process route", width="large")})
        if st.form_submit_button("Save volumes", type="primary", icon=":material/save:"):
            prods = {r.Product: dict(ss().products[r.Product], volume=int(r.Volume or 0)) for r in ed.itertuples()}
            commit(products=prods)
            st.toast("Volumes saved", icon=":material/check:")
            st.rerun()

    flows = current_flows()
    nodes = [n for n in C.DOCKS[:1] + C.MACHINES + C.DOCKS[1:] if any(n in k for k in flows)]
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("From–to matrix")
        if flows:
            mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
            for (a, b), v in flows.items():
                mat.loc[a, b] = v
            st.dataframe(mat.style.format(lambda v: f"{v:,.0f}" if v else "").background_gradient(cmap="Blues", axis=None),
                         width="stretch")
            st.caption("Rows = from, columns = to. Includes an 8% inspection → assembly rework loop. Stations missing "
                       "from the layout are skipped in the route.")
        else:
            st.info("No material flow — add machines and docks in the Layout editor.", icon=":material/info:")
    with right:
        st.subheader("Busiest links")
        if flows:
            top = pd.DataFrame([dict(Link=f"{a} → {b}", Loads=v) for (a, b), v in
                                sorted(flows.items(), key=lambda kv: -kv[1])][:8])
            st.dataframe(top, hide_index=True, width="stretch",
                         column_config={"Loads": st.column_config.ProgressColumn(
                             "Loads / shift", format="%.0f", min_value=0, max_value=float(top.Loads.max()))})


def page_optimize():
    header("Step 4", "Optimize & export", "Find an arrangement that minimises total material travel "
           "(loads × aisle distance) while keeping clearances, docks, columns and locked machines fixed.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
        clearance = c1.number_input("Minimum aisle clearance (m)", 0.0, 3.0, 1.0, 0.25)
        tol = c2.number_input("Position tolerance (m)", 0.0, 1.0, 0.3 if ss().from_detection else 0.0, 0.1,
                              help="Extra spacing that absorbs measurement error in imported plans.")
        effort = c3.segmented_control("Search effort", list(EFFORT), default="Balanced",
                                      help="Quick ≈ seconds · Balanced ≈ 10–20 s · Thorough ≈ 1 min") or "Balanced"
        run = st.button("Run optimization", type="primary", icon=":material/play_arrow:", width="stretch")

    problem, layout = build_problem(clearance, tol)
    if run:
        if not layout:
            st.error("There are no movable machines. Unlock or add machines in the Layout editor.",
                     icon=":material/error:")
        else:
            seeds, iters, refine = EFFORT[effort]
            try:
                with st.spinner(f"Evaluating arrangements ({seeds} search runs)..."):
                    t0 = time.perf_counter()
                    R = optimize(problem, layout, n_seeds=seeds, iters=iters, refine_iters=refine)
                    dt = time.perf_counter() - t0
                before, after = R["initial_cost_path"], R["final_cost_path"]
                saving = 100 * (1 - after / before) if before and np.isfinite(before) and before > 0 else 0.0
                ss().result = dict(rev=ss().rev, R=R, problem=problem, before=dict(layout), seconds=dt,
                                   saving_pct=saving, clearance=clearance, tolerance=tol, effort=effort)
            except RuntimeError as exc:
                st.error(f"{exc}. Reduce the clearance or the number of objects.", icon=":material/error:")

    res = ss().result
    if res is None:
        st.info("Choose your settings and press **Run optimization**.", icon=":material/lightbulb:")
        return
    if res["rev"] != ss().rev:
        st.warning("The layout or volumes changed since the last run. Run the optimization again to refresh "
                   "these results.", icon=":material/history:")
        return

    R, P, before = res["R"], res["problem"], res["before"]
    after = R["final_layout"]
    c0, c1 = R["initial_cost_path"], R["final_cost_path"]
    if c0 >= BIG or c1 >= BIG:
        st.warning("Some stations cannot be reached through the aisles — a dock or machine is boxed in. "
                   "Distances include a penalty.", icon=":material/warning:")

    k = st.columns(4)
    k[0].metric("Current travel", fmt(c0), "load·m per shift", delta_color="off", delta_arrow="off", border=True)
    k[1].metric("Optimized travel", fmt(c1), f"{-res['saving_pct']:+.1f}%", delta_color="inverse", border=True)
    k[2].metric("Saved per shift", fmt(c0 - c1), "load·m", delta_color="off", delta_arrow="off", border=True)
    k[3].metric("Constraints", "Pass" if R["final_feasible"] else "Fail",
                f"computed in {res['seconds']:.1f}s", delta_color="off", delta_arrow="off", border=True)
    if not R["initial_feasible"]:
        st.caption("The current layout breaks some clearance rules, so its travel figure is for reference only.")

    l, r = st.columns(2, gap="medium")
    with l.container(border=True):
        show(figure(P, before, "Current layout"))
    with r.container(border=True):
        show(figure(P, after, "Optimized layout · red = busiest routes", paths=3))

    rows = []
    for n, old in before.items():
        new = after[n]
        move = float(np.hypot(old[0] + old[2] / 2 - new[0] - new[2] / 2, old[1] + old[3] / 2 - new[1] - new[3] / 2))
        rows.append({"Equipment": n, "From x": old[0], "From y": old[1], "To x": new[0], "To y": new[1],
                     "Width": new[2], "Depth": new[3], "Rotated": abs(new[2] - old[2]) > 1e-6, "Move (m)": move})
    moves = pd.DataFrame(rows).sort_values("Move (m)", ascending=False)
    st.subheader("Relocation plan")
    st.dataframe(moves, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f")
                                for c in moves.columns if c not in ("Equipment", "Rotated")})

    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Relocation plan (CSV)", moves.to_csv(index=False), "relocation_plan.csv", "text/csv",
                       icon=":material/table:", width="stretch")
    e2.download_button("Layout drawing (PNG)", fig_png(figure(P, after, "Optimized layout", paths=3)),
                       "optimized_layout.png", "image/png", icon=":material/image:", width="stretch")
    payload = dict(app=f"{APP_NAME} {VERSION}", floor=dict(width_m=P.floor_w, depth_m=P.floor_h),
                   settings=dict(clearance_m=res["clearance"], tolerance_m=res["tolerance"], effort=res["effort"]),
                   travel_load_m_per_shift=dict(current=c0, optimized=c1, saving_pct=res["saving_pct"]),
                   layout={k: list(v) for k, v in after.items()}, fixed={k: list(v) for k, v in P.fixed.items()},
                   columns=[list(o) for o in P.obstacles], flows={f"{a} -> {b}": v for (a, b), v in P.flows.items()})
    e3.download_button("Full project (JSON)", json.dumps(payload, indent=2), "layout_project.json",
                       "application/json", icon=":material/data_object:", width="stretch")
    st.caption("Travel is measured along aisles on a 0.5 m grid. Recommendations should be reviewed against site "
               "safety, egress and utility requirements before relocation.")


# ============================================================================ navigation
init_state()
st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "icon.svg"), size="large")
PAGES = {
    "pages/dashboard": st.Page(page_dashboard, title="Dashboard", icon=":material/space_dashboard:", default=True),
    "pages/import": st.Page(page_import, title="Import floor plan", icon=":material/upload_file:", url_path="import"),
    "pages/editor": st.Page(page_editor, title="Layout editor", icon=":material/edit_square:", url_path="editor"),
    "pages/flow": st.Page(page_flow, title="Material flow", icon=":material/swap_horiz:", url_path="flow"),
    "pages/optimize": st.Page(page_optimize, title="Optimize & export", icon=":material/auto_awesome:",
                              url_path="optimize"),
}
nav = st.navigation({"Workspace": list(PAGES.values())})

with st.sidebar:
    st.caption(f"**Project** · {ss().source}")
    sc = ss().scene
    st.caption(f"{sc.floor_w:g} × {sc.floor_h:g} m · {len(sc.layout)} machines · {len(sc.fixed)} docks · "
               f"{len(sc.obstacles)} column{'s' * (len(sc.obstacles) != 1)}")
    if result_is_current():
        st.caption(f":green[● Optimized — {ss().result['saving_pct']:.1f}% less travel]")
    elif ss().result is not None:
        st.caption(":orange[● Results out of date]")

nav.run()
st.markdown(f'<div class="footer">{APP_NAME} {VERSION}</div>', unsafe_allow_html=True)
