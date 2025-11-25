"""Tests for oh_no_mcp_server package initialization."""

import oh_no_mcp_server


def test_version():
    """Test that version is defined."""
    assert hasattr(oh_no_mcp_server, "__version__")
    assert oh_no_mcp_server.__version__ == "0.1.0"
