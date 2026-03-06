from typing import List, Literal
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent_type: Literal["chat", "research"] = Field(
        description="Classify as 'chat' for greetings/pleasantries, or 'research' for questions requiring document retrieval."
    )
    fast_reply: str = Field(
        description="If 'chat', provide a quick, friendly response. If 'research', leave empty.",
        default=""
    )

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )
    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )
    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )

class DocumentGrader(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Indicates if the document provides information relevant to the user question."
    )