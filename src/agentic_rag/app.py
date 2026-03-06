import sys
import os
from loguru import logger

# Add src to python path to support modular structure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentic_rag.ui.css import custom_css
from agentic_rag.ui.gradio_app import create_gradio_ui

def main():
    """Main entry point for the Agentic RAG application."""
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    
    from typing import Dict, Any
    try:
        demo = create_gradio_ui()
        logger.info("Launching RAG Assistant UI...")
        
        launch_kwargs: Dict[str, Any] = {"css": custom_css}
        if hasattr(demo, "theme_override"):
            launch_kwargs["theme"] = getattr(demo, "theme_override")
            
        demo.launch(**launch_kwargs)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()