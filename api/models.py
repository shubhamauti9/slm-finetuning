"""
Pydantic Models for API Request/Response
OpenAI-compatible API format
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# =============================================================================
# Chat Completion Models (OpenAI-compatible)
# =============================================================================

class ChatMessage(BaseModel):
    """A single message in the chat conversation."""
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message sender"
    )
    content: str = Field(..., description="The content of the message")


class ChatCompletionRequest(BaseModel):
    """Request body for chat completion endpoint."""
    model: str = Field(default="broking-llama-7b", description="Model to use")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    max_tokens: int = Field(default=512, ge=1, le=4096, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response body for chat completion endpoint."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = "broking-llama-7b"
    choices: List[ChatCompletionChoice]
    usage: Usage


# =============================================================================
# Text Completion Models
# =============================================================================

class CompletionRequest(BaseModel):
    """Request body for text completion endpoint."""
    model: str = Field(default="broking-llama-7b", description="Model to use")
    prompt: str = Field(..., description="The prompt to complete")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = Field(default=False)


class CompletionChoice(BaseModel):
    """A single completion choice."""
    text: str
    index: int
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionResponse(BaseModel):
    """Response body for text completion endpoint."""
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str = "broking-llama-7b"
    choices: List[CompletionChoice]
    usage: Usage


# =============================================================================
# Trade Recommendation Models
# =============================================================================

class StockInfo(BaseModel):
    """Stock information for trade recommendation."""
    symbol: str = Field(..., description="Stock symbol (e.g., RELIANCE, TCS)")
    current_price: Optional[float] = Field(None, description="Current price in INR")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")
    high_52w: Optional[float] = Field(None, description="52-week high")
    low_52w: Optional[float] = Field(None, description="52-week low")
    volume: Optional[str] = Field(None, description="Trading volume")
    additional_info: Optional[str] = Field(None, description="Any additional context")


class TradeRecommendationRequest(BaseModel):
    """Request body for trade recommendation endpoint."""
    stock: StockInfo
    investment_horizon: Literal["intraday", "short_term", "medium_term", "long_term"] = "medium_term"
    risk_tolerance: Literal["low", "moderate", "high"] = "moderate"


class TradeRecommendation(BaseModel):
    """Trade recommendation response."""
    symbol: str
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target_price: Optional[float]
    risk_level: str
    analysis: str
    disclaimer: str = "This is AI-generated advice for educational purposes only. Consult a SEBI-registered advisor before investing."


class TradeRecommendationResponse(BaseModel):
    """Response body for trade recommendation endpoint."""
    success: bool
    recommendation: TradeRecommendation
    model: str = "broking-llama-7b"


# =============================================================================
# Health Check Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_name: str
    gpu_available: bool
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ModelInfo(BaseModel):
    """Model information response."""
    id: str = "broking-llama-7b"
    object: str = "model"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    owned_by: str = "local"
