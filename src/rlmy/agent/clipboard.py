"""
Purpose: Clipboard image paste utilities for RLMY.
Usage: Press Ctrl+\\ to paste image from clipboard inline.
Key Components:
    format_file_size() - human-readable file sizes
    generate_next_available_id() - short random IDs
    save_clipboard_image_inline() - main save function
    RLMY_IMAGE_DIR - module constant for temp directory
Conventions:
    - Platform-aware defaults (Windows vs Unix)
    - Overridable via RLMY_IMAGE_DIR env var
    - Fail-fast on invalid configuration
"""

import os
import random
import string
import uuid
from pathlib import Path
from PIL import ImageGrab, Image as PILImage


def _get_image_directory() -> Path:
    """
    Get the image directory with platform-aware defaults.
    
    Returns:
        Path to directory for saving clipboard images.
        
    Raises:
        RuntimeError: If RLMY_IMAGE_DIR parent doesn't exist.
    """
    env_val = os.getenv("RLMY_IMAGE_DIR")
    if env_val:
        path = Path(env_val)
        if not path.parent.exists():
            raise RuntimeError(
                f"RLMY_IMAGE_DIR parent doesn't exist: {path.parent}"
            )
        return path
    
    # Platform-aware defaults
    if os.name == "nt":  # Windows
        import tempfile
        return Path(tempfile.gettempdir()) / "rlmy"
    
    # Unix (Linux/macOS)
    return Path("/tmp/rlmy")


# Module-level constant: read environment once at import time
RLMY_IMAGE_DIR = _get_image_directory()


def format_file_size(bytes_size: int) -> str:
    """
    Format byte size as human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Human-readable string (e.g., "20KB", "2.3MB")
    """
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.0f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f}GB"


def generate_next_available_id(
    base_dir: Path,
    prefix: str = "imgpaste-",
    suffix: str = ".png"
) -> str:
    """
    Generate short random filename with collision detection.
    
    Starts with 2-character random IDs (62² = 3,844 combinations).
    Increments length on collision. Falls back to UUID if needed.
    
    Args:
        base_dir: Directory to check for collisions
        prefix: Filename prefix (default: "imgpaste-")
        suffix: Filename suffix (default: ".png")
        
    Returns:
        Filename string (e.g., "imgpaste-aB.png")
    """
    chars = string.ascii_letters + string.digits
    length = 2
    
    # Try random IDs with increasing length
    while length < 10:
        for _ in range(100):  # 100 attempts per length
            img_id = ''.join(random.choices(chars, k=length))
            filename = f"{prefix}{img_id}{suffix}"
            if not (base_dir / filename).exists():
                return filename
        length += 1
    
    # Fallback to UUID if random collisions persist
    return f"{prefix}{uuid.uuid4().hex[:8]}{suffix}"


def save_clipboard_image_inline(temp_dir: Path) -> tuple[str | None, str]:
    """
    Save clipboard image to disk and return inline text for insertion.
    
    Args:
        temp_dir: Directory to save image (for dependency injection)
        
    Returns:
        Tuple of (inline_text, error_message).
        On success: ("🖼️ /path (WxH, SIZE)", "")
        On failure: (None, "error description")
    """
    try:
        image = ImageGrab.grabclipboard()
    except AttributeError:
        # Linux without xclip/xsel
        return None, "Image paste requires xclip on Linux (apt install xclip)"
    
    if image is None:
        return None, "No image in clipboard"
    
    if not isinstance(image, PILImage.Image):
        return None, f"Clipboard contains {type(image).__name__}, not an image"
    
    # Ensure directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    filename = generate_next_available_id(temp_dir)
    save_path = temp_dir / filename
    
    # Save image
    try:
        image.save(save_path, format="PNG")
    except Exception as e:
        return None, f"Failed to save image: {e}"
    
    # Get image info
    width, height = image.size
    file_size = save_path.stat().st_size
    size_str = format_file_size(file_size)
    
    # Return inline text
    inline_text = f"🖼️ {save_path} ({width}x{height}, {size_str})"
    return inline_text, ""
