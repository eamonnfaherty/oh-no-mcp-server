"""Tests for oh_no_mcp_server.server module."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, AsyncMock
from mcp.types import TextContent

from oh_no_mcp_server.server import (
    app,
    read_file_content,
    read_directory_recursive,
    list_tools,
    call_tool,
    list_prompts,
    get_prompt,
    main,
    PERFORMANCE_REVIEW_PROMPT,
)


class TestReadFileContent:
    """Tests for read_file_content function."""

    def test_read_file_success(self, tmp_path):
        """Test reading a file successfully."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        result = read_file_content(str(test_file))
        assert result == test_content

    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        result = read_file_content("/nonexistent/file.txt")
        assert "Error reading file" in result
        assert "/nonexistent/file.txt" in result

    def test_read_file_permission_error(self):
        """Test reading a file with permission error."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = read_file_content("/protected/file.txt")
            assert "Error reading file" in result
            assert "Access denied" in result


class TestReadDirectoryRecursive:
    """Tests for read_directory_recursive function."""

    def test_read_directory_success(self, tmp_path):
        """Test reading a directory successfully."""
        # Create test files
        (tmp_path / "file1.py").write_text("print('file1')")
        (tmp_path / "file2.py").write_text("print('file2')")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.py").write_text("print('file3')")

        result = read_directory_recursive(str(tmp_path))

        assert "file1.py" in result
        assert "file2.py" in result
        assert "subdir/file3.py" in result or "subdir\\file3.py" in result
        assert result["file1.py"] == "print('file1')"
        assert result["file2.py"] == "print('file2')"

    def test_read_directory_not_exists(self):
        """Test reading a non-existent directory."""
        result = read_directory_recursive("/nonexistent/directory")
        assert "error" in result
        assert "does not exist" in result["error"]

    def test_read_directory_not_a_directory(self, tmp_path):
        """Test reading a file instead of directory."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        result = read_directory_recursive(str(test_file))
        assert "error" in result
        assert "is not a directory" in result["error"]

    def test_read_directory_excludes_patterns(self, tmp_path):
        """Test that excluded patterns are skipped."""
        # Create files in excluded directories
        excluded_dirs = ["__pycache__", ".git", "node_modules", ".venv"]
        for excluded_dir in excluded_dirs:
            excluded = tmp_path / excluded_dir
            excluded.mkdir()
            (excluded / "file.py").write_text("excluded")

        # Create a valid file
        (tmp_path / "valid.py").write_text("valid content")

        result = read_directory_recursive(str(tmp_path))

        assert "valid.py" in result
        # Ensure excluded files are not present
        for excluded_dir in excluded_dirs:
            assert not any(excluded_dir in key for key in result.keys())

    def test_read_directory_skips_binary_files(self, tmp_path):
        """Test that binary files are skipped."""
        # Create a binary file with invalid UTF-8
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\xff\xfe\xfd\xfc")

        # Create a text file
        text_file = tmp_path / "text.txt"
        text_file.write_text("text content")

        result = read_directory_recursive(str(tmp_path))

        assert "text.txt" in result
        assert "binary.bin" not in result

    def test_read_directory_handles_permission_error(self, tmp_path):
        """Test that permission errors are handled gracefully."""
        # Create a readable file
        (tmp_path / "readable.py").write_text("readable")

        # Mock a file that raises PermissionError
        original_open = open

        def mock_open_with_permission_error(file, *args, **kwargs):
            if "unreadable" in str(file):
                raise PermissionError("Access denied")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_with_permission_error):
            # Create the unreadable file path
            (tmp_path / "unreadable.py").touch()

            result = read_directory_recursive(str(tmp_path))

            # Should have the readable file but not the unreadable one
            assert "readable.py" in result
            assert "unreadable.py" not in result


@pytest.mark.asyncio
class TestListTools:
    """Tests for list_tools function."""

    async def test_list_tools(self):
        """Test that list_tools returns the correct tool definition."""
        tools = await list_tools()

        assert len(tools) == 1
        assert tools[0].name == "oh_no"
        assert "code performance" in tools[0].description.lower()
        assert "scope" in tools[0].inputSchema["properties"]
        assert "content" in tools[0].inputSchema["properties"]
        assert "output_path" in tools[0].inputSchema["properties"]
        assert tools[0].inputSchema["required"] == ["scope", "content"]


@pytest.mark.asyncio
class TestCallTool:
    """Tests for call_tool function."""

    async def test_call_tool_unknown_tool(self):
        """Test calling an unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("unknown_tool", {})

    async def test_call_tool_missing_required_arguments(self):
        """Test calling tool without required arguments raises ValueError."""
        with pytest.raises(ValueError, match="required"):
            await call_tool("oh_no", {})

        with pytest.raises(ValueError, match="required"):
            await call_tool("oh_no", {"scope": "text"})

    async def test_call_tool_invalid_scope(self):
        """Test calling tool with invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            await call_tool(
                "oh_no",
                {"scope": "invalid", "content": "test"}
            )

    async def test_call_tool_text_scope(self):
        """Test calling tool with text scope."""
        result = await call_tool(
            "oh_no",
            {"scope": "text", "content": "def foo(): pass"}
        )

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "def foo(): pass" in result[0].text
        assert "text output" in result[0].text

    async def test_call_tool_file_scope(self, tmp_path):
        """Test calling tool with file scope."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        result = await call_tool(
            "oh_no",
            {"scope": "file", "content": str(test_file)}
        )

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "print('hello')" in result[0].text
        assert str(test_file) in result[0].text

    async def test_call_tool_file_scope_error(self):
        """Test calling tool with file scope for non-existent file."""
        result = await call_tool(
            "oh_no",
            {"scope": "file", "content": "/nonexistent/file.py"}
        )

        assert len(result) == 1
        assert "Error reading file" in result[0].text

    async def test_call_tool_directory_scope_missing_output_path(self):
        """Test calling tool with directory scope without output_path raises ValueError."""
        with pytest.raises(ValueError, match="output_path is required"):
            await call_tool(
                "oh_no",
                {"scope": "directory", "content": "/some/dir"}
            )

    async def test_call_tool_directory_scope_success(self, tmp_path):
        """Test calling tool with directory scope."""
        # Create test files
        (tmp_path / "file1.py").write_text("print('file1')")
        (tmp_path / "file2.py").write_text("print('file2')")

        output_path = tmp_path / "report.md"

        result = await call_tool(
            "oh_no",
            {
                "scope": "directory",
                "content": str(tmp_path),
                "output_path": str(output_path)
            }
        )

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "file1.py" in result[0].text
        assert "file2.py" in result[0].text
        assert "print('file1')" in result[0].text
        assert "print('file2')" in result[0].text
        assert str(output_path) in result[0].text

    async def test_call_tool_directory_scope_error(self):
        """Test calling tool with directory scope for non-existent directory."""
        result = await call_tool(
            "oh_no",
            {
                "scope": "directory",
                "content": "/nonexistent/dir",
                "output_path": "/tmp/report.md"
            }
        )

        assert len(result) == 1
        assert "does not exist" in result[0].text


@pytest.mark.asyncio
class TestListPrompts:
    """Tests for list_prompts function."""

    async def test_list_prompts(self):
        """Test that list_prompts returns the correct prompt definition."""
        prompts = await list_prompts()

        assert len(prompts) == 1
        assert prompts[0].name == "oh_no"
        assert "performance" in prompts[0].description.lower()
        assert len(prompts[0].arguments) == 3
        assert prompts[0].arguments[0].name == "scope"
        assert prompts[0].arguments[0].required is True
        assert prompts[0].arguments[1].name == "content"
        assert prompts[0].arguments[1].required is True
        assert prompts[0].arguments[2].name == "output_path"
        assert prompts[0].arguments[2].required is False


@pytest.mark.asyncio
class TestGetPrompt:
    """Tests for get_prompt function."""

    async def test_get_prompt_unknown_prompt(self):
        """Test getting an unknown prompt raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prompt"):
            await get_prompt("unknown_prompt", {})

    async def test_get_prompt_no_arguments(self):
        """Test getting prompt without arguments raises ValueError."""
        with pytest.raises(ValueError, match="Arguments are required"):
            await get_prompt("oh_no", None)

    async def test_get_prompt_missing_required_arguments(self):
        """Test getting prompt without required arguments raises ValueError."""
        with pytest.raises(ValueError, match="required"):
            await get_prompt("oh_no", {})

        with pytest.raises(ValueError, match="required"):
            await get_prompt("oh_no", {"scope": "text"})

    async def test_get_prompt_invalid_scope(self):
        """Test getting prompt with invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            await get_prompt(
                "oh_no",
                {"scope": "invalid", "content": "test"}
            )

    async def test_get_prompt_text_scope(self):
        """Test getting prompt with text scope."""
        result = await get_prompt(
            "oh_no",
            {"scope": "text", "content": "def bar(): return 42"}
        )

        assert result.description == "Performance review for text: def bar(): return 42"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert "def bar(): return 42" in result.messages[0].content.text
        assert "text output" in result.messages[0].content.text

    async def test_get_prompt_file_scope(self, tmp_path):
        """Test getting prompt with file scope."""
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1 + 1")

        result = await get_prompt(
            "oh_no",
            {"scope": "file", "content": str(test_file)}
        )

        assert "Performance review for file" in result.description
        assert len(result.messages) == 1
        assert "x = 1 + 1" in result.messages[0].content.text
        assert str(test_file) in result.messages[0].content.text

    async def test_get_prompt_directory_scope_missing_output_path(self):
        """Test getting prompt with directory scope without output_path raises ValueError."""
        with pytest.raises(ValueError, match="output_path is required"):
            await get_prompt(
                "oh_no",
                {"scope": "directory", "content": "/some/dir"}
            )

    async def test_get_prompt_directory_scope_success(self, tmp_path):
        """Test getting prompt with directory scope."""
        # Create test files
        (tmp_path / "main.py").write_text("import sys")
        subdir = tmp_path / "lib"
        subdir.mkdir()
        (subdir / "util.py").write_text("def util(): pass")

        output_path = "/tmp/review.md"

        result = await get_prompt(
            "oh_no",
            {
                "scope": "directory",
                "content": str(tmp_path),
                "output_path": output_path
            }
        )

        assert "Performance review for directory" in result.description
        assert len(result.messages) == 1
        assert "main.py" in result.messages[0].content.text
        assert "import sys" in result.messages[0].content.text
        assert output_path in result.messages[0].content.text

    async def test_get_prompt_directory_scope_error(self):
        """Test getting prompt with directory scope for non-existent directory."""
        result = await get_prompt(
            "oh_no",
            {
                "scope": "directory",
                "content": "/nonexistent/path",
                "output_path": "/tmp/out.md"
            }
        )

        assert len(result.messages) == 1
        assert "does not exist" in result.messages[0].content.text


class TestMain:
    """Tests for main function."""

    @pytest.mark.asyncio
    async def test_main_runs(self):
        """Test that main function runs without errors."""
        # Mock stdio_server and app.run
        mock_read_stream = Mock()
        mock_write_stream = Mock()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_stdio_server():
            yield mock_read_stream, mock_write_stream

        with patch("oh_no_mcp_server.server.stdio_server", return_value=mock_stdio_server()):
            with patch.object(app, "run", new_callable=AsyncMock) as mock_run:
                with patch.object(app, "create_initialization_options", return_value=Mock()):
                    await main()

                    # Verify that app.run was called
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args[0]
                    assert call_args[0] == mock_read_stream
                    assert call_args[1] == mock_write_stream

    def test_module_executable(self):
        """Test that the module's __main__ block is covered by using runpy."""
        import runpy
        from contextlib import asynccontextmanager
        import sys

        # Save original __name__
        original_name = sys.modules.get("oh_no_mcp_server.server").__name__ if "oh_no_mcp_server.server" in sys.modules else None

        # Mock the dependencies
        mock_read_stream = Mock()
        mock_write_stream = Mock()

        @asynccontextmanager
        async def mock_stdio_server():
            yield mock_read_stream, mock_write_stream

        # Patch dependencies before running as __main__
        with patch("oh_no_mcp_server.server.stdio_server", return_value=mock_stdio_server()):
            with patch("oh_no_mcp_server.server.asyncio.run") as mock_asyncio_run:
                # Temporarily modify __name__ to trigger the if block
                import oh_no_mcp_server.server as server_module
                original_module_name = server_module.__name__
                try:
                    server_module.__name__ = "__main__"
                    # Re-execute the module code to trigger the if __name__ == "__main__" block
                    exec(compile(open(server_module.__file__).read(), server_module.__file__, 'exec'),
                         server_module.__dict__)
                    # Verify asyncio.run was called with main
                    mock_asyncio_run.assert_called_once()
                finally:
                    server_module.__name__ = original_module_name


class TestConstants:
    """Tests for module-level constants."""

    def test_performance_review_prompt_format(self):
        """Test that PERFORMANCE_REVIEW_PROMPT has required placeholders."""
        assert "{code_content}" in PERFORMANCE_REVIEW_PROMPT
        assert "{output_instructions}" in PERFORMANCE_REVIEW_PROMPT
        assert "performance" in PERFORMANCE_REVIEW_PROMPT.lower()

    def test_app_name(self):
        """Test that app has correct name."""
        assert app.name == "oh-no-mcp-server"
