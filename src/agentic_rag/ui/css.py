custom_css = """
    /* ============================================
       PREMIUM OVERRIDES FOR GRADIO 4.x/6.x THEME
       Enforcing Clean, Modern Light Mode
       ============================================ */
    
    .gradio-container { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif !important;
        background-color: #f8fafc !important; 
    }
    
    /* Clean up the main header with a sleek gradient and typography */
    .header-title {
        text-align: center;
        margin-bottom: 2.5rem !important;
        margin-top: 1.5rem !important;
    }

    .header-title h1 {
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.025em !important;
        background: linear-gradient(135deg, #1e293b 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }

    /* Refine File UI components: softer borders and subtle hover effects */
    .file-preview {
        background: #ffffff !important;
        border: 1px dashed #cbd5e1 !important; 
        border-radius: 12px !important;
        transition: all 0.2s ease-in-out;
    }
    .file-preview:hover {
        border-color: #3b82f6 !important;
        background: #f0fdf4 !important; /* Extremely light green tint on hover */
    }
    
    /* Chat UI Enhancement: Premium framing */
    .chatbot {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02) !important;
    }

    /* Message Bubbles - Modern Apple-like aesthetic with delicate shadows */
    .message.user {
        background-color: #2563eb !important; /* Solid vibrant blue */
        color: #ffffff !important; 
        border: none !important;
        border-radius: 18px 18px 4px 18px !important; /* Asymmetric bubble */
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.15), 0 2px 4px -1px rgba(37, 99, 235, 0.1) !important;
        padding: 12px 16px !important;
        font-weight: 400 !important;
    }
    
    .message.bot {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        color: #1e293b !important;
        border-radius: 18px 18px 18px 4px !important; /* Asymmetric bubble */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
        padding: 12px 16px !important;
        line-height: 1.6 !important;
    }

    /* Input area styling */
    .gradio-container textarea {
        border-radius: 20px !important;
        padding-left: 16px !important;
    }

    /* Hide the footer watermark */
    footer {
        visibility: hidden;
    }
"""