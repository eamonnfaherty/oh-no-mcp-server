#!/usr/bin/env python3
"""MCP server for code performance reviews."""

import os
import asyncio
from pathlib import Path
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Prompt, PromptArgument, GetPromptResult, PromptMessage


app = Server("oh-no-mcp-server")


PERFORMANCE_REVIEW_PROMPT = """Please review the following code for performance issues and potential optimizations:

{code_content}

Provide a detailed analysis including:
1. Performance bottlenecks or inefficiencies
2. Memory usage concerns
3. Algorithm complexity issues
4. Suggestions for optimization
5. Best practices recommendations

{output_instructions}"""


def read_file_content(file_path: str) -> str:
    """Read and return file content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


def read_directory_recursive(directory: str) -> dict[str, str]:
    """Recursively read all files in a directory."""
    files_content = {}
    dir_path = Path(directory)

    if not dir_path.exists():
        return {"error": f"Directory {directory} does not exist"}

    if not dir_path.is_dir():
        return {"error": f"{directory} is not a directory"}

    # Common patterns to exclude
    exclude_patterns = {
        '.git', '.svn', '.hg', '__pycache__', 'node_modules',
        '.venv', 'venv', 'dist', 'build', '.egg-info'
    }

    for file_path in dir_path.rglob('*'):
        # Skip if any parent directory matches exclude patterns
        if any(part in exclude_patterns for part in file_path.parts):
            continue

        if file_path.is_file():
            # Skip binary and very large files
            try:
                # Try to read as text
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    relative_path = file_path.relative_to(dir_path)
                    files_content[str(relative_path)] = content
            except (UnicodeDecodeError, PermissionError):
                # Skip binary files or files we can't read
                continue

    return files_content


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="oh_no",
            description="Get a prompt for reviewing code performance. Supports highlighted text, single files, or entire directories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["text", "file", "directory"],
                        "description": "The scope of the review: 'text' for highlighted code, 'file' for a single file, 'directory' for recursive directory scan"
                    },
                    "content": {
                        "type": "string",
                        "description": "For scope='text': the code snippet. For scope='file': the file path. For scope='directory': the directory path"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "For scope='directory': where to write the report. Optional for other scopes."
                    }
                },
                "required": ["scope", "content"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    if name != "oh_no":
        raise ValueError(f"Unknown tool: {name}")

    scope = arguments.get("scope")
    content = arguments.get("content")
    output_path = arguments.get("output_path")

    if not scope or not content:
        raise ValueError("Both 'scope' and 'content' are required")

    code_content = ""
    output_instructions = ""

    if scope == "text":
        # Direct code snippet
        code_content = f"```\n{content}\n```"
        output_instructions = "Please provide the review as text output."

    elif scope == "file":
        # Single file
        file_content = read_file_content(content)
        code_content = f"File: {content}\n\n```\n{file_content}\n```"
        output_instructions = "Please provide the review as text output."

    elif scope == "directory":
        # Directory - recursive scan
        if not output_path:
            raise ValueError("output_path is required when scope='directory'")

        files_content = read_directory_recursive(content)

        if "error" in files_content:
            return [TextContent(type="text", text=files_content["error"])]

        # Build combined content
        code_parts = [f"Directory: {content}\n\n"]
        for file_path, file_content in files_content.items():
            code_parts.append(f"File: {file_path}\n```\n{file_content}\n```\n\n")

        code_content = "".join(code_parts)
        output_instructions = f"Please provide a comprehensive review and write the complete report to: {output_path}"
    else:
        raise ValueError(f"Invalid scope: {scope}. Must be 'text', 'file', or 'directory'")

    # Generate the prompt
    prompt = PERFORMANCE_REVIEW_PROMPT.format(
        code_content=code_content,
        output_instructions=output_instructions
    )

    return [TextContent(type="text", text=prompt)]


@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="oh_no",
            description="Review code for performance issues and optimizations",
            arguments=[
                PromptArgument(
                    name="scope",
                    description="The scope of review: 'text', 'file', or 'directory'",
                    required=True
                ),
                PromptArgument(
                    name="content",
                    description="The code text, file path, or directory path to review",
                    required=True
                ),
                PromptArgument(
                    name="output_path",
                    description="For directories: where to write the report",
                    required=False
                )
            ]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Get a prompt by name."""
    if name != "oh_no":
        raise ValueError(f"Unknown prompt: {name}")

    if not arguments:
        raise ValueError("Arguments are required")

    scope = arguments.get("scope")
    content = arguments.get("content")
    output_path = arguments.get("output_path")

    if not scope or not content:
        raise ValueError("Both 'scope' and 'content' are required")

    code_content = ""
    output_instructions = ""

    if scope == "text":
        code_content = f"```\n{content}\n```"
        output_instructions = "Please provide the review as text output."

    elif scope == "file":
        file_content = read_file_content(content)
        code_content = f"File: {content}\n\n```\n{file_content}\n```"
        output_instructions = "Please provide the review as text output."

    elif scope == "directory":
        if not output_path:
            raise ValueError("output_path is required when scope='directory'")

        files_content = read_directory_recursive(content)

        if "error" in files_content:
            code_content = files_content["error"]
            output_instructions = ""
        else:
            code_parts = [f"Directory: {content}\n\n"]
            for file_path, file_content in files_content.items():
                code_parts.append(f"File: {file_path}\n```\n{file_content}\n```\n\n")

            code_content = "".join(code_parts)
            output_instructions = f"Please provide a comprehensive review and write the complete report to: {output_path}"
    else:
        raise ValueError(f"Invalid scope: {scope}")

    prompt_text = PERFORMANCE_REVIEW_PROMPT.format(
        code_content=code_content,
        output_instructions=output_instructions
    )

    return GetPromptResult(
        description=f"Performance review for {scope}: {content}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
    )


async def async_main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
