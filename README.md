# LLaMA 7B Fine-tuning for Broking Domain

Fine-tuned LLaMA 7B using Unsloth for stock market analysis and trade recommendations, deployed with FastAPI.

## Quick Start

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Unsloth requires CUDA. Ensure you have a compatible GPU and CUDA toolkit installed.

### 2. Fine-tune the Model

```powershell
cd d:\rag-git\fine-tuning-test
python -m training.train
```

Training will:
- Load LLaMA 2 7B (4-bit quantized)
- Apply LoRA adapters
- Train on broking domain data
- Save model to `models/broking-llama-7b/`

### 3. Start the API Server

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload for development:
```powershell
python -m api.main
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |
| `/v1/completions` | POST | Text completion |
| `/v1/trade/recommend` | POST | Trade recommendation |

### Example: Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Analyze RELIANCE stock for investment"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Example: Trade Recommendation

```bash
curl -X POST http://localhost:8000/v1/trade/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "stock": {
      "symbol": "HDFC",
      "current_price": 1650,
      "pe_ratio": 18.5
    },
    "investment_horizon": "medium_term",
    "risk_tolerance": "moderate"
  }'
```

## Project Structure

```
fine-tuning-test/
├── data/
│   └── broking_training_data.json   # Training dataset
├── training/
│   ├── config.py                     # Training configuration
│   └── train.py                      # Training script
├── models/                           # Saved models
├── api/
│   ├── main.py                       # FastAPI application
│   ├── models.py                     # Pydantic models
│   └── inference.py                  # Model inference
├── requirements.txt
└── README.md
```

## Configuration

### Training Config (`training/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `unsloth/llama-2-7b-bnb-4bit` | Base model |
| `lora_r` | 16 | LoRA rank |
| `learning_rate` | 2e-4 | Learning rate |
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Batch size |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Auto-detect | Path to model |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Hardware Requirements

### Training
- GPU: NVIDIA with 8GB+ VRAM (RTX 3060/4060 or better)
- RAM: 16GB+
- Storage: 20GB for model + data

### Inference
- GPU: NVIDIA with 6GB+ VRAM (4-bit quantized)
- RAM: 8GB+

## Development

### Testing the API

```powershell
# Health check
curl http://localhost:8000/health

# Test chat
curl -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

### API Documentation

OpenAPI docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT License
