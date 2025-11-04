import streamlit as st
import json
from pathlib import Path
from typing import List, Any, Dict


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file and return a list of JSON objects."""
    traces = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            traces.append(json.loads(line))
    return traces


def format_arguments(arguments: Any) -> str:
    """Format arguments for display."""
    if arguments is None:
        return "None"
    if isinstance(arguments, str):
        try:
            # Try to parse as JSON for pretty formatting
            parsed = json.loads(arguments)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except:
            return arguments
    return json.dumps(arguments, indent=2, ensure_ascii=False)


def format_output(output: Any) -> str:
    """Format output for display."""
    if output is None:
        return "No output"
    if isinstance(output, list):
        if len(output) == 0:
            return "Empty list"
        # Check if it's a list of dictionaries with 'text' field
        if isinstance(output[0], dict) and 'text' in output[0]:
            return '\n\n'.join([item.get('text', '') for item in output if item.get('text')])
        return json.dumps(output, indent=2, ensure_ascii=False)
    if isinstance(output, str):
        # Check if it's a JSON string
        try:
            parsed = json.loads(output)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except:
            return output
    return json.dumps(output, indent=2, ensure_ascii=False)


def display_trace_entry(entry: Dict[str, Any], trace_uid: str, selected_trace_index: int, entry_index: int, display_index: int, action_type: str):
    """Display a single trace entry in an expandable box."""
    
    # Create unique keys that include trace UID to ensure proper updates across files
    base_key = f"{trace_uid}_entry_{entry_index}_disp_{display_index}"
    
    if action_type == "reasoning":
        with st.expander(
            f"🧠 Reasoning #{display_index+1} (Trace Index: {entry_index})", 
            expanded=False
        ):
            output = entry.get('output', [])
            formatted_output = format_output(output)
            
            # Calculate height based on content length (max 600px)
            content_height = min(len(formatted_output.split('\n')) * 20, 600)
            content_height = max(content_height, 100)
            
            # Use code block for better formatting of long content
            st.markdown("**Reasoning Output:**")
            st.text_area(
                "Content:",
                value=formatted_output,
                height=int(content_height),
                key=f"text_reasoning_{base_key}",
                disabled=True,
                label_visibility="collapsed"
            )
    
    elif action_type == "tool_call":
        tool_name = entry.get('tool_name', 'Unknown Tool')
        with st.expander(
            f"🔧 Tool Call #{display_index+1}: {tool_name} (Trace Index: {entry_index})", 
            expanded=False
        ):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Arguments:**")
                arguments = entry.get('arguments', None)
                formatted_args = format_arguments(arguments)
                
                # Calculate height based on content length (max 600px)
                args_height = min(len(formatted_args.split('\n')) * 20, 600)
                args_height = max(args_height, 100)
                
                st.text_area(
                    "Content:",
                    value=formatted_args,
                    height=int(args_height),
                    key=f"text_args_{base_key}",
                    disabled=True,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("**Output:**")
                output = entry.get('output', None)
                formatted_output = format_output(output)
                
                # Calculate height based on content length (max 600px)
                output_height = min(len(formatted_output.split('\n')) * 20, 600)
                output_height = max(output_height, 100)
                
                st.text_area(
                    "Content:",
                    value=formatted_output,
                    height=int(output_height),
                    key=f"text_output_{base_key}",
                    disabled=True,
                    label_visibility="collapsed"
                )


@st.cache_data
def load_traces_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and cache traces from a file."""
    return read_jsonl(file_path)


@st.cache_data
def load_query_mapping() -> Dict[str, str]:
    """Load query_id to query mapping from the decrypted dataset."""
    mapping_file = Path('data/browsecomp_plus_decrypted.jsonl')
    if not mapping_file.exists():
        return {}
    
    id2query = {}
    for record in read_jsonl(str(mapping_file)):
        query_id = record.get('query_id')
        query = record.get('query')
        if query_id and query:
            id2query[str(query_id)] = query
    return id2query


def main():
    st.set_page_config(
        page_title="Agent Trace Viewer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Agent Trace Viewer")
    st.markdown("Visualize agent traces with reasoning and tool calls")
    
    # Initialize session state
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if 'trace_index' not in st.session_state:
        st.session_state.trace_index = 0
    
    # Sidebar for file selection
    st.sidebar.header("Trace File Selection")
    
    # Load traces from data directory
    data_dir = Path('data/decrypted_run_files')
    
    if not data_dir.exists():
        st.error("Data directory not found. Please ensure 'data/decrypted_run_files' exists.")
        return
    
    # Get all subdirectories and their trace files
    trace_files = {}
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for jsonl_file in subdir.glob("*.jsonl"):
                display_name = f"{subdir.name}/{jsonl_file.name}"
                trace_files[display_name] = str(jsonl_file)
    
    if not trace_files:
        st.warning("No trace files found in data/decrypted_run_files")
        return
    
    # File selector
    selected_file = st.sidebar.selectbox(
        "Select a trace file:",
        options=list(trace_files.keys()),
        key="file_selector",
        index=0
    )
    
    # Load traces
    file_path = trace_files[selected_file]
    
    # Check if file changed
    if st.session_state.file_path != file_path:
        st.session_state.file_path = file_path
        st.session_state.trace_index = 0
    
    traces = load_traces_from_file(file_path)
    
    if not traces:
        st.warning("No traces found in selected file")
        return
    
    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Total traces:** {len(traces)}")
    
    # Trace selector with proper key
    selected_index = st.sidebar.selectbox(
        "Select a trace:",
        options=list(range(len(traces))),
        format_func=lambda x: f"Trace #{x+1}",
        key="trace_selector",
        index=st.session_state.trace_index
    )
    
    # Update session state
    st.session_state.trace_index = selected_index
    
    selected_trace = traces[selected_index]
    
    # Create a unique identifier for this trace (file + index)
    trace_uid = f"{selected_file}_{selected_index}"
    
    # Load query mapping
    id2query = load_query_mapping()
    
    # Get query text
    query_id = selected_trace.get('query_id', 'Unknown')
    query_text = None
    
    # Try to get query from the mapping
    if query_id:
        query_text = id2query.get(str(query_id))
    
    # Fallback to query field in trace if mapping doesn't have it
    if not query_text:
        query_text = selected_trace.get('query')
    
    # Display trace metadata
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Query ID", query_id)
    
    with col2:
        st.metric("Search Counts", selected_trace.get('search_counts', 'N/A'))
    
    with col3:
        st.metric("Status", selected_trace.get('status', 'N/A'))
    
    # Display query information
    st.markdown("### Query:")
    if query_text:
        st.info(query_text)
    else:
        st.warning(f"Query text not available (Query ID: {query_id})")
    
    # Display trace entries
    st.markdown("---")
    st.markdown("### Trace Entries:")
    
    result = selected_trace.get('result', [])
    if not result:
        st.warning("No trace entries found")
        return
    
    # Debug information
    with st.expander("🔍 Debug Information", expanded=False):
        st.write(f"**Selected Trace Index:** {selected_index}")
        st.write(f"**Selected Trace Data:**")
        st.json({
            "query_id": selected_trace.get('query_id', 'N/A'),
            "status": selected_trace.get('status', 'N/A'),
            "search_counts": selected_trace.get('search_counts', 'N/A'),
            "total_entries": len(result)
        })
        st.write(f"**Entry Types:**")
        entry_types = [entry.get('type', 'unknown') for entry in result]
        st.write(f"Reasoning: {entry_types.count('reasoning')}, Tool Calls: {entry_types.count('tool_call')}")
    
    # Filter buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        show_reasoning = st.checkbox("Show Reasoning", value=True, key=f"show_reasoning_{trace_uid}")
    with col2:
        show_tool_calls = st.checkbox("Show Tool Calls", value=True, key=f"show_tool_calls_{trace_uid}")
    with col3:
        st.write(f"Total Entries: {len(result)}")
    
    # Wrap trace entries in a container with unique key to force re-render
    trace_container = st.container()
    with trace_container:
        # Display each trace entry
        displayed_count = 0
        for i, entry in enumerate(result):
            entry_type = entry.get('type', 'unknown')
            
            if entry_type == "reasoning" and not show_reasoning:
                continue
            if entry_type == "tool_call" and not show_tool_calls:
                continue
            
            # Display with trace UID, selected trace index, entry index, and display index
            display_trace_entry(entry, trace_uid, selected_index, i, displayed_count, entry_type)
            displayed_count += 1


if __name__ == "__main__":
    main()

