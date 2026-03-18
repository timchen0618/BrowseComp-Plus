import re
import json
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Styling ───────────────────────────────────────────────────────────────────

COLUMN_ACCENT_COLORS = [
    "#4F8EF7",  # blue
    "#E07B3F",  # orange
    "#3DB87A",  # green
    "#A259F7",  # purple
    "#E05C7B",  # pink
    "#3BBDBD",  # teal
]

CSS = """
<style>
/* Lock the page to the viewport — no outer scroll */
html, body {
    overflow: hidden !important;
    height: 100% !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main {
    overflow: hidden !important;
    height: 100vh !important;
}
.block-container {
    overflow: hidden !important;
    height: 100vh !important;
    padding-top: 2rem !important;
    padding-bottom: 0 !important;
}

/* Query card */
.query-card {
    background: #1E1E2E;
    border-left: 4px solid #7C83FD;
    border-radius: 6px;
    padding: 16px 20px;
    margin-top: 4px;
    margin-bottom: 16px;
    color: #CDD6F4;
    font-size: 1.0rem;
    line-height: 1.65;
}
.query-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.query-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7C83FD;
}
.query-id-badge {
    background: #313244;
    color: #CBA6F7;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.06em;
}

/* Plan column card */
.plan-card {
    border-radius: 8px;
    border: 1px solid #313244;
    background: #181825;
    height: 100%;
    overflow: hidden;
}
.plan-header {
    padding: 10px 16px 8px 16px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #313244;
}
.plan-body {
    padding: 14px 16px;
    color: #CDD6F4;
    font-size: 0.92rem;
    line-height: 1.75;
    word-wrap: break-word;
    overflow-y: auto;
    /* total viewport minus: top padding + title/caption + nav row + query card + plan header + margins */
    height: calc(100vh - 360px);
}
.plan-missing {
    color: #585B70;
    font-style: italic;
    padding: 14px 16px;
    font-size: 0.88rem;
}

/* Navigation bar */
.nav-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1E1E2E;
    border-radius: 6px;
    padding: 8px 16px;
    margin-bottom: 18px;
}
.nav-count {
    color: #585B70;
    font-size: 0.82rem;
    margin-left: auto;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: #181825;
}
section[data-testid="stSidebar"] label {
    color: #A6ADC8 !important;
    font-size: 0.82rem !important;
}
</style>
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_plan(output: str) -> str:
    if not isinstance(output, str):
        return str(output)
    m = re.search(r"<plan>(.*?)</plan>", output, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else output.strip()


@st.cache_data
def load_plan_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id", ""))
            records[qid] = {
                "query_id": qid,
                "query_text": obj.get("query_text", ""),
                "plan": extract_plan(obj.get("output", "")),
            }
    return records


def scan_jsonl_files(directory: str) -> Dict[str, str]:
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        return {}
    return {f.name: str(f) for f in sorted(p.glob("*.jsonl"))}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_query_card(qid: str, query_text: str) -> None:
    text = html_escape(query_text) if query_text else "<em>(query text not available)</em>"
    st.markdown(
        f"""
        <div class="query-card">
            <div class="query-card-header">
                <span class="query-label">Question</span>
                <span class="query-id-badge">ID&nbsp;&nbsp;{qid}</span>
            </div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plan_card(label: str, plan_text: Optional[str], accent: str) -> None:
    header = f'<div class="plan-header" style="color:{accent}; background:rgba(0,0,0,0.15);">{label}</div>'
    if plan_text is None:
        body = '<div class="plan-missing">No plan available for this query.</div>'
    else:
        escaped = html_escape(plan_text)
        # Convert both literal '\n' sequences and real newline characters to <br>
        escaped = escaped.replace('\\n', '<br>').replace('\n', '<br>')
        body = f'<div class="plan-body">{escaped}</div>'
    st.markdown(
        f'<div class="plan-card">{header}{body}</div>',
        unsafe_allow_html=True,
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Plan Viewer",
        page_icon="📋",
        layout="wide",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 Plan Viewer")
        st.markdown("---")

        data_dir = st.text_input(
            "Plans directory",
            value="data/plans",
            help="Directory with one .jsonl file per model.",
        )

        all_files = scan_jsonl_files(data_dir)

        if not all_files:
            st.warning(f"No `.jsonl` files found in `{data_dir}`.")
            st.stop()

        selected_names = st.multiselect(
            "Models to compare",
            options=list(all_files.keys()),
            default=list(all_files.keys()),
        )

        if not selected_names:
            st.info("Select at least one model.")
            st.stop()

    # ── Load files ────────────────────────────────────────────────────────────
    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for name in selected_names:
        loaded[name] = load_plan_file(all_files[name])

    # Intersection: only queries present in every loaded file
    record_sets = [set(r.keys()) for r in loaded.values()]
    common = record_sets[0].intersection(*record_sets[1:])
    try:
        all_qids = sorted(common, key=lambda x: int(x))
    except ValueError:
        all_qids = sorted(common)

    if not all_qids:
        st.warning("No queries found.")
        st.stop()

    # Helper: get query text from any loaded file
    def get_query_text(qid: str) -> str:
        for records in loaded.values():
            if qid in records and records[qid]["query_text"]:
                return records[qid]["query_text"]
        return ""

    # ── Query selector in sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")

        if "qid_index" not in st.session_state:
            st.session_state.qid_index = 0
        st.session_state.qid_index = min(st.session_state.qid_index, len(all_qids) - 1)

        jump_labels = [
            f"[{qid}]  {get_query_text(qid)[:60]}{'…' if len(get_query_text(qid)) > 60 else ''}"
            for qid in all_qids
        ]

        # Buttons come BEFORE the selectbox so that st.rerun() fires before
        # the selectbox widget is instantiated, allowing session state to be set.
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("← Prev", use_container_width=True, disabled=st.session_state.qid_index == 0):
                st.session_state.qid_index -= 1
                st.session_state["query_selector"] = jump_labels[st.session_state.qid_index]
                st.rerun()
        with next_col:
            if st.button("Next →", use_container_width=True, disabled=st.session_state.qid_index == len(all_qids) - 1):
                st.session_state.qid_index += 1
                st.session_state["query_selector"] = jump_labels[st.session_state.qid_index]
                st.rerun()

        chosen_label = st.selectbox(
            f"Query  ({len(all_qids)} total)",
            options=jump_labels,
            index=st.session_state.qid_index,
            key="query_selector",
        )
        jumped = jump_labels.index(chosen_label)
        if jumped != st.session_state.qid_index:
            st.session_state.qid_index = jumped
            st.rerun()

    # ── Query card ────────────────────────────────────────────────────────────
    selected_qid = all_qids[st.session_state.qid_index]
    render_query_card(selected_qid, get_query_text(selected_qid))

    # ── Side-by-side plan columns ─────────────────────────────────────────────
    cols = st.columns(len(selected_names))
    for i, (col, name) in enumerate(zip(cols, selected_names)):
        with col:
            records = loaded[name]
            plan_text = records[selected_qid]["plan"] if selected_qid in records else None
            accent = COLUMN_ACCENT_COLORS[i % len(COLUMN_ACCENT_COLORS)]
            render_plan_card(
                label=name.removesuffix(".jsonl"),
                plan_text=plan_text,
                accent=accent,
            )


if __name__ == "__main__":
    main()
