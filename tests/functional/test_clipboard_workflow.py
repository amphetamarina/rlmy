"""
Functional tests for clipboard image paste workflow.

Tests the critical path: clipboard → save → file exists → PIL can read.
This is integration-style testing focused on catching real bugs in the workflow.
"""

import re
import pytest
from pathlib import Path
from PIL import Image as PILImage

from rlmy.agent.clipboard import (
    format_file_size,
    generate_next_available_id,
    save_clipboard_image_inline,
)


def test_format_file_size():
    """Test human-readable file size formatting."""
    assert format_file_size(500) == "500B"
    assert format_file_size(1024) == "1KB"
    assert format_file_size(2048) == "2KB"
    assert format_file_size(1024 * 1024) == "1.0MB"
    assert format_file_size(int(2.5 * 1024 * 1024)) == "2.5MB"
    assert format_file_size(1024 * 1024 * 1024) == "1.0GB"
    assert format_file_size(int(1.5 * 1024 * 1024 * 1024)) == "1.5GB"


def test_generate_next_available_id_no_collision(tmp_path):
    """Test ID generation when no files exist."""
    filename = generate_next_available_id(tmp_path)
    
    # Should start with prefix
    assert filename.startswith("imgpaste-")
    
    # Should have 2-char ID by default (imgpaste-XX.png = 16 chars)
    assert len(filename) == len("imgpaste-XX.png")
    
    # Should end with .png
    assert filename.endswith(".png")


def test_generate_next_available_id_with_collision(tmp_path):
    """Test ID generation handles collisions by trying new IDs."""
    # Create first file
    first_id = generate_next_available_id(tmp_path)
    (tmp_path / first_id).touch()
    
    # Second ID should be different
    second_id = generate_next_available_id(tmp_path)
    assert second_id != first_id
    assert not (tmp_path / second_id).exists()


def test_save_clipboard_image_functional(tmp_path, monkeypatch):
    """
    CRITICAL PATH TEST: Save image → file exists → PIL can read.
    
    This is a functional test, not unit test. It tests:
    - Image save to disk
    - File exists at expected path
    - PIL can read saved file (vision_query will depend on this)
    """
    # Create test image (simulates clipboard)
    test_img = PILImage.new('RGB', (100, 100), color='red')
    
    # Mock clipboard to return our test image
    def mock_grabclipboard():
        return test_img
    
    monkeypatch.setattr(
        "rlmy.agent.clipboard.ImageGrab.grabclipboard",
        mock_grabclipboard
    )
    
    # ACT: Save image
    inline_text, error = save_clipboard_image_inline(tmp_path)
    
    # ASSERT: Success (no error)
    assert error == "", f"Save failed: {error}"
    assert inline_text is not None
    assert "🖼️" in inline_text
    
    # ASSERT: Extract path from inline text
    # Format: "🖼️ /path/to/imgpaste-XX.png (100x100, 20KB)"
    match = re.search(r'(\S+\.png)', inline_text)
    assert match, f"No path found in inline text: {inline_text}"
    
    # Get just the filename from the full path
    saved_filename = match.group(1).split('/')[-1]
    saved_path = tmp_path / saved_filename
    
    # ASSERT: File exists on disk
    assert saved_path.exists(), f"File not saved: {saved_path}"
    
    # ASSERT: PIL can read the file (critical for vision_query)
    with PILImage.open(saved_path) as img:
        assert img.size == (100, 100)
        assert img.mode == 'RGB'
    
    # ASSERT: Inline text contains correct dimensions
    assert "(100x100," in inline_text
    
    # ASSERT: File size is reasonable
    assert saved_path.stat().st_size > 0


def test_multiple_pastes_no_collision(tmp_path, monkeypatch):
    """Test multiple pastes generate different IDs."""
    test_img = PILImage.new('RGB', (50, 50), color='blue')
    
    def mock_grabclipboard():
        return test_img
    
    monkeypatch.setattr(
        "rlmy.agent.clipboard.ImageGrab.grabclipboard",
        mock_grabclipboard
    )
    
    # Save 3 images
    paths = []
    for _ in range(3):
        inline_text, error = save_clipboard_image_inline(tmp_path)
        assert error == "", f"Save failed: {error}"
        
        # Extract filename
        match = re.search(r'(\S+\.png)', inline_text)
        assert match
        saved_filename = match.group(1).split('/')[-1]
        paths.append(tmp_path / saved_filename)
    
    # All paths should be different
    assert len(set(str(p) for p in paths)) == 3, "Generated duplicate filenames"
    
    # All files should exist
    for p in paths:
        assert p.exists(), f"File not created: {p}"


def test_save_clipboard_image_no_image_in_clipboard(tmp_path, monkeypatch):
    """Test error handling when clipboard is empty."""
    def mock_grabclipboard():
        return None
    
    monkeypatch.setattr(
        "rlmy.agent.clipboard.ImageGrab.grabclipboard",
        mock_grabclipboard
    )
    
    inline_text, error = save_clipboard_image_inline(tmp_path)
    
    assert inline_text is None
    assert "No image in clipboard" in error


def test_save_clipboard_image_wrong_type_in_clipboard(tmp_path, monkeypatch):
    """Test error handling when clipboard contains non-image data."""
    def mock_grabclipboard():
        return "some text"  # Not an image
    
    monkeypatch.setattr(
        "rlmy.agent.clipboard.ImageGrab.grabclipboard",
        mock_grabclipboard
    )
    
    inline_text, error = save_clipboard_image_inline(tmp_path)
    
    assert inline_text is None
    assert "not an image" in error
