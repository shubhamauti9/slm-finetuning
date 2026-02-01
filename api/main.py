"""
FastAPI Server for Broking LLaMA Model
OpenAI-compatible API endpoints
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    TradeRecommendationRequest,
    TradeRecommendationResponse,
    TradeRecommendation,
    HealthResponse,
    ModelInfo,
    Usage,
)
from api.inference import model_manager


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    print("\n" + "=" * 60)
    print("Starting Broking LLaMA API Server")
    print("=" * 60)
    
    # Load model
    model_path = os.environ.get("MODEL_PATH", None)
    success = model_manager.load_model(model_path)
    
    if not success:
        print("[WARNING] Model not loaded. API will work in limited mode.")
    
    print("\nServer is ready!")
    print("=" * 60 + "\n")
    
    yield
    
    # Shutdown
    print("\nShutting down server...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Broking LLaMA API",
    description="Fine-tuned LLaMA 7B for stock market analysis and trade recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    import torch
    from datetime import datetime
    
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "degraded",
        model_loaded=model_manager.model_loaded,
        model_name=model_manager.model_path or "none",
        gpu_available=torch.cuda.is_available(),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/v1/models", tags=["Models"])
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id="broking-llama-7b",
                object="model",
                owned_by="local",
            ).model_dump()
        ]
    }


@app.get("/v1/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(model_id: str):
    """Get model information."""
    if model_id != "broking-llama-7b":
        raise HTTPException(status_code=404, detail="Model not found")
    return ModelInfo(id=model_id)


# =============================================================================
# Chat Completion Endpoints
# =============================================================================

@app.post("/v1/chat/completions", tags=["Chat"])
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    
    Supports both streaming and non-streaming responses.
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )
    
    # Format chat messages into prompt
    prompt = model_manager.format_chat_prompt(
        [msg.model_dump() for msg in request.messages]
    )
    
    if request.stream:
        return StreamingResponse(
            stream_chat_response(prompt, request),
            media_type="text/event-stream",
        )
    
    # Non-streaming response
    try:
        response_text = model_manager.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        # Count tokens
        prompt_tokens = model_manager.count_tokens(prompt)
        completion_tokens = model_manager.count_tokens(response_text)
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
            ),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_response(
    prompt: str, 
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream chat completion response."""
    import uuid
    from datetime import datetime
    
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    try:
        for chunk in model_manager.generate_stream(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(data)}\n\n"
        
        # Send final chunk
        final_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# =============================================================================
# Text Completion Endpoints
# =============================================================================

@app.post("/v1/completions", response_model=CompletionResponse, tags=["Completions"])
async def create_completion(request: CompletionRequest):
    """Create a text completion."""
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )
    
    try:
        response_text = model_manager.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        
        prompt_tokens = model_manager.count_tokens(request.prompt)
        completion_tokens = model_manager.count_tokens(response_text)
        
        return CompletionResponse(
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text,
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
            ),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trade Recommendation Endpoint
# =============================================================================

@app.post("/v1/trade/recommend", response_model=TradeRecommendationResponse, tags=["Trading"])
async def get_trade_recommendation(request: TradeRecommendationRequest):
    """
    Get a trade recommendation for a specific stock.
    
    This endpoint formats the stock information into a prompt and returns
    a structured trade recommendation.
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )
    
    # Build stock info string
    stock_info_parts = [f"Symbol: {request.stock.symbol}"]
    if request.stock.current_price:
        stock_info_parts.append(f"Current Price: ₹{request.stock.current_price}")
    if request.stock.pe_ratio:
        stock_info_parts.append(f"P/E Ratio: {request.stock.pe_ratio}")
    if request.stock.high_52w and request.stock.low_52w:
        stock_info_parts.append(f"52-week range: ₹{request.stock.low_52w}-{request.stock.high_52w}")
    if request.stock.volume:
        stock_info_parts.append(f"Volume: {request.stock.volume}")
    if request.stock.additional_info:
        stock_info_parts.append(request.stock.additional_info)
    
    stock_info = ", ".join(stock_info_parts)
    
    # Create prompt
    from training.config import ALPACA_PROMPT
    
    prompt = ALPACA_PROMPT.format(
        instruction=f"Analyze the stock {request.stock.symbol} and provide a trading recommendation for {request.investment_horizon} investment with {request.risk_tolerance} risk tolerance.",
        input=stock_info,
        output="",
    )
    
    try:
        response_text = model_manager.generate(
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Parse response into structured format
        # This is a simplified parser - production would be more robust
        recommendation = TradeRecommendation(
            symbol=request.stock.symbol,
            recommendation="HOLD",  # Default, would parse from response
            entry_price=request.stock.current_price,
            stop_loss=request.stock.current_price * 0.92 if request.stock.current_price else None,
            target_price=request.stock.current_price * 1.12 if request.stock.current_price else None,
            risk_level=request.risk_tolerance.upper(),
            analysis=response_text,
        )
        
        return TradeRecommendationResponse(
            success=True,
            recommendation=recommendation,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
            }
        },
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,  # Disable in production
    )
