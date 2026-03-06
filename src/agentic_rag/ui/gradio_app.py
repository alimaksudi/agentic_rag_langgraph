import gradio as gr
from loguru import logger
from typeguard import typechecked

from agentic_rag.core.chat_interface import ChatInterface
from agentic_rag.core.document_manager import DocumentManager
from agentic_rag.core.rag_system import RAGSystem
from agentic_rag.db.vector_db_manager import VectorDbManager
from agentic_rag.db.parent_store_manager import ParentStoreManager
from agentic_rag.document_chunker import DocumentChuncker
from agentic_rag.exceptions import RAGSystemError, AgentExecutionError
from agentic_rag.config import settings


@typechecked
def create_gradio_ui() -> gr.Blocks:
    """Creates and configures the Gradio user interface."""
    logger.info("Initializing UI components and dependency graph...")
    
    try:
        # Dependency Injection Initialization
        vector_db = VectorDbManager()
        parent_store = ParentStoreManager()
        chunker = DocumentChuncker()
        
        rag_system = RAGSystem(
            vector_db=vector_db,
            parent_store=parent_store,
            chunker=chunker
        )
        rag_system.initialize()
    except RAGSystemError as e:
        logger.critical(f"Failed to initialize core RAG infrastructure: {e}")
        # In a real app, we might return a degraded UI here. For now, we raise to halt startup.
        raise
    
    doc_manager = DocumentManager(rag_system)
    chat_interface = ChatInterface(rag_system)
    
    def format_file_list() -> str:
        files = doc_manager.get_markdown_files()
        if not files:
            return "No documents available in the knowledge base"
        return "\n".join([f"{f}" for f in files])
    
    def upload_handler(files, progress=gr.Progress()):
        if not files:
            return None, format_file_list()
            
        added, skipped = doc_manager.add_documents(
            files, 
            progress_callback=lambda p, desc: progress(p, desc=desc)
        )
        
        gr.Info(f"Added: {added} | Skipped: {skipped}")
        return None, format_file_list()
    
    def clear_handler():
        doc_manager.clear_all()
        gr.Info(f"Removed all documents")
        return format_file_list()
    
    async def chat_handler(msg, hist):
        try:
            async for chunk in chat_interface.chat_stream(msg, hist):
                yield chunk
        except AgentExecutionError as e:
            yield f"Service Error: The agent encountered an issue processing your request."
        except Exception as e:
            logger.error(f"Chat UI Error: {e}")
            yield f"An unexpected error occurred."
    
    def clear_chat_handler():
        chat_interface.clear_session()
    
    # Define an enterprise-grade, clean light theme
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_md
    ).set(
        body_background_fill="#f8fafc", # Very soft slate-50
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e2e8f0",
        block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)",
        border_color_primary="#3b82f6",
        color_accent_soft="#eff6ff",
        slider_color="#3b82f6",
        block_title_text_color="#1e293b",
        block_title_text_weight="600",
        block_label_text_color="#475569",
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
        button_primary_shadow="0 4px 6px -1px rgba(37, 99, 235, 0.2), 0 2px 4px -1px rgba(37, 99, 235, 0.1)",
        button_secondary_background_fill="#f1f5f9",
        button_secondary_background_fill_hover="#e2e8f0",
        button_secondary_text_color="#334155",
        button_secondary_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    )

    with gr.Blocks(title="Agentic RAG Assistant", fill_height=True) as demo:
        gr.Markdown("# Agentic RAG Assistant", elem_classes=["header-title"])
        
        with gr.Tab("Documents"):
            gr.Markdown("Build your local knowledge base by uploading PDF or Markdown files.")
            
            with gr.Row():
                # Left Column: Ingestion
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Upload & Index")
                    files_input = gr.File(
                        label="Drop files here",
                        file_count="multiple",
                        type="filepath",
                        height=200
                    )
                    
                    with gr.Row():
                        add_btn = gr.Button("Index Documents", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear KB", variant="stop", size="lg")
                        
                    # Adding a status box for better UX
                    status_box = gr.Textbox(
                        label="Indexing Status",
                        value="Ready.",
                        interactive=False,
                        lines=2
                    )

                # Right Column: Knowledge Base Contents
                with gr.Column(scale=2):
                    gr.Markdown("### 2. Available Knowledge")
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh View", size="sm")
                        
                    file_list = gr.Textbox(
                        value=format_file_list(),
                        interactive=False,
                        lines=12,
                        label="Indexed Documents",
                        elem_id="file-list-box"
                    )
            
            # Event Bindings for Documents
            def upload_with_status(files, progress=gr.Progress()):
                if not files:
                    return None, format_file_list(), "No files selected."
                    
                added, skipped = doc_manager.add_documents(
                    files, 
                    progress_callback=lambda p, desc: progress(p, desc=desc)
                )
                
                status_msg = f"Successfully added {added} document(s). Skipped {skipped} duplicates."
                return None, format_file_list(), status_msg
                
            def clear_with_status():
                doc_manager.clear_all()
                return format_file_list(), "Knowledge Base cleared."

            add_btn.click(
                upload_with_status, 
                [files_input], 
                [files_input, file_list, status_box], 
                show_progress="minimal"
            )
            refresh_btn.click(format_file_list, None, file_list)
            clear_btn.click(clear_with_status, None, [file_list, status_box])
        
        with gr.Tab("Reasoning & Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        placeholder="Ask me anything about your documents! I can perform deep, autonomous research.",
                        show_label=False
                    )
                    chatbot.clear(clear_chat_handler)
                    
                    gr.ChatInterface(fn=chat_handler, chatbot=chatbot)
                
                # Contextual Settings sidebar
                with gr.Column(scale=1):
                    with gr.Accordion("Agent Controls", open=True):
                        # Dynamic provider generation
                        provider_display = settings.ACTIVE_LLM_CONFIG.upper()
                        if provider_display == "OLLAMA":
                            provider_display = "Local Privacy Mode (Ollama)"
                        else:
                            provider_display = f"Cloud Provider ({provider_display})"
                            
                        gr.Markdown(f"""
                        **Current Execution Profile**: 
                        - Provider: {provider_display}
                        - Semantic Parent-Child Retrieval Enabled
                        - Context Compression Threshold: {settings.BASE_TOKEN_THRESHOLD} tokens
                        """)
                        # Placeholders for future dynamic controls
                        gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Temperature (Context Grounding)", interactive=False)
                        gr.Slider(minimum=1, maximum=15, value=8, step=1, label="Max Research Cycles", interactive=False)
                        
                        gr.Markdown("*Configurations are currently governed by `.env` to ensure application stability during open-source distribution.*")
    
    # Store theme for app.py to pass into launch()
    demo.theme_override = custom_theme
    return demo  # type: ignore