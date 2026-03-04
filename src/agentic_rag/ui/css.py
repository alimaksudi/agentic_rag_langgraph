custom_css = """
    /* ============================================
       MINIMAL OVERRIDES FOR GRADIO 4.x/6.x THEME
       Enforcing Clean Light Mode
       ============================================ */
    
    .gradio-container { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
    }
    
    /* Clean up the main header */
    .header-title {
        text-align: center;
        margin-bottom: 2rem !important;
        margin-top: 1rem !important;
        color: #1e293b; /* Slate 800 */
    }

    /* Refine File UI components slightly for light mode */
    .file-preview {
        background: #f8fafc !important; /* Slate 50 */
        border-color: #e2e8f0 !important; /* Slate 200 */
    }
    
    /* Ensure the Chat UI doesn't look overly boxed in \\*/
    .chatbot {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
    }

    /* Message Bubbles - Clean Light Look */
    .message.user {
        background-color: #eff6ff !important; /* Blue 50 */
        border: 1px solid #bfdbfe !important; /* Blue 200 */
        color: #1e40af !important; /* Blue 800 */
    }
    
    .message.bot {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important; /* Slate 700 */
    }

    /* Hide the footer watermark */
    footer {
        visibility: hidden;
    }
"""