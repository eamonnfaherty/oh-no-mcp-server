# Oh No MCP Server

An MCP (Model Context Protocol) server that provides code performance review capabilities. This server exposes tools and prompts to analyze code for performance issues, bottlenecks, and optimization opportunities.

## Features

- **Highlighted Text Review**: Analyze code snippets selected in your IDE
- **Single File Review**: Analyze an entire file for performance issues
- **Directory Review**: Recursively scan and analyze all files in a directory
- **Flexible Output**: Text output for snippets/files, written reports for directories

## Installation

### Using uvx (Recommended)

For quick installation and usage:

```bash
uvx --from git+https://github.com/eamonnfaherty/oh-no-mcp-server oh-no-mcp
```

### From Source

1. Clone this repository:

```bash
git clone https://github.com/eamonnfaherty/oh-no-mcp-server.git
cd oh-no-mcp-server
```

2. Install dependencies using `uv`:

```bash
make install
```

For development with all extras:

```bash
make install-dev
```

## Usage

### As an MCP Server

Run the server using the Makefile:

```bash
make run
```

Or directly with `uv` using the console script:

```bash
uv run oh-no-mcp
```

Or with `uvx` (no installation needed):

```bash
uvx --from git+https://github.com/eamonnfaherty/oh-no-mcp-server oh-no-mcp
```

### Configuration for Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

Using `uvx` with git repository (recommended):

```json
{
  "mcpServers": {
    "oh-no": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/eamonnfaherty/oh-no-mcp-server",
        "oh-no-mcp"
      ]
    }
  }
}
```

Or using `uv` from a local clone:

```json
{
  "mcpServers": {
    "oh-no": {
      "command": "uv",
      "args": ["--directory", "/path/to/oh-no-mcp-server", "run", "oh-no-mcp"]
    }
  }
}
```

## Available Tools

### `oh_no`

Returns a prompt for reviewing code performance.

**Parameters:**
- `scope` (required): Type of review - "text", "file", or "directory"
- `content` (required): The code text, file path, or directory path
- `output_path` (optional): For directories, where to write the report

**Examples:**

```python
# Review highlighted text
{
  "scope": "text",
  "content": "def slow_function():\n    result = []\n    for i in range(1000):\n        result.append(i)"
}

# Review a single file
{
  "scope": "file",
  "content": "/path/to/your/file.py"
}

# Review a directory
{
  "scope": "directory",
  "content": "/path/to/your/project",
  "output_path": "/path/to/output/performance_report.md"
}
```

## Available Prompts

### `oh_no`

Directly invokes a performance review with the same parameters as the tool.

**Arguments:**
- `scope`: "text", "file", or "directory"
- `content`: The code, file path, or directory path
- `output_path`: (For directories) Where to write the report

## What the Review Includes

The performance review analyzes code for:

1. **Performance Bottlenecks**: Identifies slow operations and inefficiencies
2. **Memory Usage**: Highlights potential memory leaks or excessive allocations
3. **Algorithm Complexity**: Analyzes time and space complexity
4. **Optimization Suggestions**: Provides actionable improvements
5. **Best Practices**: Recommends industry-standard patterns

## Project Structure

```
oh-no-mcp-server/
├── pyproject.toml
├── README.md
└── src/
    └── oh_no_mcp_server/
        ├── __init__.py
        └── server.py
```

## Development

This project uses `uv` for dependency management. All common development tasks are available via the Makefile.

### Makefile Commands

#### Installation

- **`make install`** - Install project dependencies using `uv sync`
- **`make install-dev`** - Install dependencies with all extras for development

#### Code Quality

- **`make format`** - Auto-format code using `black` and fix issues with `ruff`
- **`make lint`** - Check code style with `ruff` and verify formatting with `black` (no changes made)

#### Testing

- **`make test`** - Run all unit tests using `pytest`
- **`make test-coverage`** - Run tests with coverage reporting (terminal + HTML)
- **`make test-coverage-report`** - Open the HTML coverage report in your browser

#### Evaluation

- **`make eval`** - Run evaluation tests with standard output
- **`make eval-verbose`** - Run evaluation tests with detailed output (includes print statements)

#### Utilities

- **`make clean`** - Remove all build artifacts, cache files, and coverage reports
  - Deletes `__pycache__` directories
  - Removes `.pyc` files
  - Cleans up `.egg-info` directories
  - Removes `.pytest_cache`, `htmlcov`, `.coverage`
  - Removes `dist` and `build` directories
- **`make run`** - Start the MCP server using `uv run oh-no-mcp`

## Testing Framework

This project includes two types of tests:

### Unit Tests (`tests/`)
Traditional unit tests with 100% code coverage testing individual functions and components.

### Evaluation Tests (`evals/`)
Advanced semantic similarity-based tests that compare MCP tool outputs using:
- **BERT embeddings** - Semantic understanding via sentence transformers
- **Self-BLEU** - N-gram overlap measurement
- **Cosine similarity** - Normalized embedding comparison

Run evaluation tests:

```bash
make eval              # Run evaluations
make eval-verbose      # Run with detailed output
```

Each evaluation provides detailed metrics and a pass/fail result. See [evals/README.md](evals/README.md) for more details.

## License

MIT
