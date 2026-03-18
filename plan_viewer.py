import re
import json
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Data loading ──────────────────────────────────────────────────────────────

def extract_plan(output: str) -> str:
    """Extract content between <plan> </plan> tags, falling back to raw output."""
    if not isinstance(output, str):
        return str(output)
    m = re.search(r"<plan>(.*?)</plan>", output, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else output.strip()


@st.cache_data
def load_plan_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load a plan JSONL file; returns {query_id: record}."""
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
                "output": obj.get("output", ""),
                "plan": extract_plan(obj.get("output", "")),
            }
    return records


def scan_jsonl_files(directory: str) -> Dict[str, str]:
    """Return {display_name: absolute_path} for all *.jsonl files in directory."""
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        return {}
    return {f.name: str(f) for f in sorted(p.glob("*.jsonl"))}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def text_area_height(text: str, min_h: int = 150, max_h: int = 800) -> int:
    lines = text.count("\n") + 1
    return max(min_h, min(lines * 22, max_h))


def render_plan_column(label: str, plan_text: Optional[str], col_key: str) -> None:
    """Render a single plan column."""
    st.markdown(f"**{label}**")
    if plan_text is None:
        st.warning("No plan available for this query.")
        return
    st.text_area(
        label="plan",
        value=plan_text,
        height=text_area_height(plan_text),
        key=col_key,
        disabled=True,
        label_visibility="collapsed",
    )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Plan Viewer",
        page_icon="📋",
        layout="wide",
    )
    st.title("📋 Plan Viewer")
    st.caption("Compare plans from multiple models for the same query.")

    # ── Sidebar: directory + file selection ───────────────────────────────────
    st.sidebar.header("Data")

    data_dir = st.sidebar.text_input(
        "Plans directory",
        value="data/plans",
        help="Path to a directory containing .jsonl plan files (one file per model).",
    )

    all_files = scan_jsonl_files(data_dir)

    if not all_files:
        st.sidebar.warning(f"No .jsonl files found in `{data_dir}`.")
        st.info(
            f"Put one `.jsonl` file per model inside **`{data_dir}/`**.\n\n"
            "Each line must have the format:\n"
            "```json\n"
            '{\"query_id\": \"1\", \"query_text\": \"...\", \"output\": \"...<plan>plan text</plan>...\"}\n'
            "```"
        )
        return

    selected_names = st.sidebar.multiselect(
        "Select files to compare",
        options=list(all_files.keys()),
        default=list(all_files.keys()),
        help="Choose which model files to show side-by-side.",
    )

    if not selected_names:
        st.info("Select at least one file in the sidebar.")
        return

    # ── Load selected files ───────────────────────────────────────────────────
    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {label: {qid: record}}
    for name in selected_names:
        loaded[name] = load_plan_file(all_files[name])

    # Build the union of query IDs (preserve insertion order, sorted numerically)
    all_qids_raw: List[str] = []
    seen: set = set()
    for records in loaded.values():
        for qid in records:
            if qid not in seen:
                all_qids_raw.append(qid)
                seen.add(qid)

    try:
        all_qids = sorted(all_qids_raw, key=lambda x: int(x))
    except ValueError:
        all_qids = sorted(all_qids_raw)

    if not all_qids:
        st.warning("No queries found in the selected files.")
        return

    # ── Sidebar: query selection ──────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("Query")

    # Get query text from whichever file has it
    def get_query_text(qid: str) -> str:
        for records in loaded.values():
            if qid in records and records[qid]["query_text"]:
                return records[qid]["query_text"]
        return ""

    query_labels = {
        qid: f"[{qid}]  {get_query_text(qid)[:80]}{'…' if len(get_query_text(qid)) > 80 else ''}"
        for qid in all_qids
    }

    selected_qid = st.sidebar.selectbox(
        "Select a query",
        options=all_qids,
        format_func=lambda qid: query_labels[qid],
        key="query_selector",
    )

    # ── Main area ─────────────────────────────────────────────────────────────
    query_text = get_query_text(selected_qid)

    st.markdown("---")
    col_meta1, col_meta2 = st.columns([1, 8])
    with col_meta1:
        st.metric("Query ID", selected_qid)
    with col_meta2:
        st.markdown("**Query**")
        st.info(query_text if query_text else "*(query text not available)*")

    st.markdown("---")
    st.markdown(f"**Comparing {len(selected_names)} plan(s)**")

    # Side-by-side columns
    cols = st.columns(len(selected_names))
    for col, name in zip(cols, selected_names):
        with col:
            records = loaded[name]
            plan_text = records[selected_qid]["plan"] if selected_qid in records else None
            render_plan_column(
                label=name,
                plan_text=plan_text,
                col_key=f"plan_{name}_{selected_qid}",
            )


if __name__ == "__main__":
    main()
