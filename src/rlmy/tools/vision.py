"""
Purpose: Vision analysis tools for multimodal agent capabilities.
Usage: Agent calls peek_image() to analyze images with vision-capable LM.
Key Components:
    peek_image() - Analyze image with specific query (cautious, not comprehensive)
Conventions:
    - Uses filesystem permission system (relative → sandbox, absolute → approval)
    - Clipboard images pre-trusted (RLMY_IMAGE_DIR in TRUSTED_ROOTS)
    - Query parameter encourages specific questions, not broad descriptions
"""

import dspy
from pathlib import Path


def peek_image(image_path: str, query: str = "Describe what you see in this image") -> str:
    """
    Analyze an image using a vision-capable model.
    
    ⚠️ IMPORTANT: This tool only returns what you specifically ask for. It doesn't provide
    comprehensive analysis unless you ask for it. Be specific in your query to get useful
    information. Think of this as "peeking" at specific aspects, not seeing everything.
    
    Args:
        image_path: Path to image file
            - Relative paths resolve to sandbox workspace
            - Absolute paths require permission (clipboard images pre-approved)
        query: Specific question about the image
            - Good: "What error message is shown in this screenshot?"
            - Good: "What UI framework is this application using?"
            - Good: "What are the main colors in this image?"
            - Avoid vague: "Tell me about this image"
    
    Returns:
        Vision model's response to your specific query
        
    Examples:
        # Analyze an error screenshot
        peek_image("screenshot.png", "What error message is displayed?")
        
        # Check pasted clipboard image
        peek_image("/tmp/rlmy/imgpaste-aB.png", "What programming language is this code?")
        
        # Analyze a chart
        peek_image("chart.png", "What is the main trend shown in this chart?")
    """
    # Import here to avoid circular dependency
    from rlmy.agent.filesystem import _require_filesystem_root, _request_filesystem_permission
    
    # Path resolution: relative → sandbox, absolute → permission check
    path = Path(image_path)
    
    if not path.is_absolute():
        # Relative path: resolve against sandbox workspace
        fs_root = _require_filesystem_root()
        path = (fs_root / path).resolve()
    else:
        # Absolute path: use existing permission system
        # (clipboard images already trusted via RLMY_IMAGE_DIR in TRUSTED_ROOTS)
        granted, reason = _request_filesystem_permission(path, "analyze image")
        if not granted:
            return f"Error: Permission denied to access {path}. {reason}"
    
    # Validate file exists and is accessible
    if not path.exists():
        return f"Error: Image not found at {path}"
    
    if not path.is_file():
        return f"Error: {path} is not a file"
    
    # Analyze image with vision-capable model
    try:
        # Define vision signature
        class ImageAnalysis(dspy.Signature):
            """Analyze the image based on the specific query provided."""
            image: dspy.Image = dspy.InputField(desc="The image to analyze")
            query: str = dspy.InputField(desc="Specific question about the image")
            response: str = dspy.OutputField(desc="Answer to the specific query")
        
        # Use predictor with the configured model (already vision-capable from config)
        analyzer = dspy.Predict(ImageAnalysis)
        
        # Load image and analyze
        image = dspy.Image(str(path))
        result = analyzer(image=image, query=query)
        
        return result.response
        
    except Exception as e:
        return f"Error analyzing image: {e}"
