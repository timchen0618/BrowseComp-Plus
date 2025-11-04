# Agent Trace Viewer

A Streamlit application for visualizing agent traces with reasoning and tool calls.

## Features

- **Expandable Boxes**: Traces are displayed in collapsible expandable boxes to save space
- **Reasoning Display**: Shows the output of reasoning actions with proper formatting
- **Tool Call Display**: Shows both arguments and actual content returned by tool calls in a side-by-side layout
- **Handles Long Output**: Automatically adjusts the height of text areas based on content length
- **Filter Controls**: Toggle visibility of reasoning and tool calls separately
- **File Navigation**: Browse through all trace files in the data directory

## Usage

### Installation

First, make sure you have the required dependencies:

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install streamlit
```

### Running the Application

```bash
streamlit run trace_viewer.py
```

The application will open in your default web browser, typically at `http://localhost:8501`.

### Using the Viewer

1. **Select a Trace File**: Use the sidebar to select from available trace files in `data/decrypted_run_files/`
2. **Select a Trace**: Choose a specific trace from the loaded file
3. **View Metadata**: See query ID, search counts, and status at the top
4. **Filter Actions**: Use the checkboxes to show/hide reasoning or tool calls
5. **Expand Entries**: Click on expandable boxes to see the details of each action
6. **Scroll**: Text areas are scrollable and automatically adjust height based on content length

## Structure

The application expects trace files in JSONL format with the following structure:

```json
{
  "query_id": "123",
  "search_counts": 5,
  "status": "completed",
  "result": [
    {
      "type": "reasoning",
      "tool_name": null,
      "arguments": null,
      "output": ["reasoning text"]
    },
    {
      "type": "tool_call",
      "tool_name": "search",
      "arguments": "{\"query\": \"example\"}",
      "output": "{...}"
    }
  ]
}
```

## Features Details

### Reasoning Display
- Shows in expandable boxes labeled with 🧠
- Handles both simple strings and complex nested structures
- Extracts text from dictionaries when present

### Tool Call Display
- Shows in expandable boxes labeled with 🔧 and the tool name
- Displays arguments on the left side
- Displays output on the right side
- Both are in scrollable text areas
- Automatically formats JSON for readability

## Troubleshooting

If you encounter issues:

1. **Data directory not found**: Ensure `data/decrypted_run_files/` exists with trace files
2. **No traces shown**: Check that your trace files are in JSONL format
3. **Streamlit not found**: Run `uv sync` or `pip install streamlit`
4. **Same trace always shown**: This was a bug that has been fixed. If you're using an old version, try refreshing the page (F5 or Cmd+R)

## Recent Fixes

- **Fixed trace selection**: Added proper session state management with unique keys for all selectboxes to ensure that changing selections actually updates the displayed trace.

