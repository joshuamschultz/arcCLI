# Arc CLI

Unified command-line interface for Arc products. Currently supports [ArcLLM](https://github.com/joshuamschultz/arcllm) for multi-provider LLM operations.

## Installation

```bash
pip install arccli
```

Development install:

```bash
pip install -e /path/to/arcllm    # dependency first
pip install -e /path/to/arccli
```

## Quick Start

```bash
# List providers and models
arc llm providers
arc llm models

# Inspect a provider
arc llm provider anthropic

# Make a call
arc llm call anthropic "What is the capital of France?"

# Validate your setup
arc llm validate
```

## Commands

| Command | Description |
|---------|-------------|
| `arc llm providers` | List all available providers |
| `arc llm provider NAME` | Show provider details and models |
| `arc llm models` | List all models across providers |
| `arc llm call PROVIDER PROMPT` | Make an LLM call |
| `arc llm config` | Show global ArcLLM configuration |
| `arc llm validate` | Validate configs and API keys |
| `arc llm version` | Show version info |

Every command supports `--json` for machine-readable output.

### Making Calls

```bash
# Basic call
arc llm call anthropic "Explain TCP in one sentence"

# Override model and parameters
arc llm call anthropic "Summarize this" \
  --model claude-haiku-4-5-20251001 \
  --temperature 0.3 \
  --max-tokens 100 \
  --system "Be concise"

# With usage details
arc llm call anthropic "Hello" --telemetry --verbose

# Full JSON response for scripting
arc llm call anthropic "Hello" --json
```

#### Module Flags

Toggle modules per call (defaults come from config):

| Flag | Description |
|------|-------------|
| `--retry` / `--no-retry` | Retry with exponential backoff |
| `--fallback` / `--no-fallback` | Fallback to alternate providers |
| `--rate-limit` / `--no-rate-limit` | Token bucket rate limiting |
| `--telemetry` / `--no-telemetry` | Cost and timing tracking |
| `--audit` / `--no-audit` | Audit metadata logging |
| `--security` / `--no-security` | PII redaction + request signing |
| `--otel` / `--no-otel` | OpenTelemetry distributed tracing |

### Filtering Models

```bash
arc llm models --provider anthropic    # by provider
arc llm models --tools                 # tool-use capable
arc llm models --vision                # vision capable
```

## Extensibility

The `arc` namespace is designed for future products:

```bash
arc run ...      # ArcRun
arc agent ...    # ArcAgent
```

## Requirements

- Python >= 3.11
- [arcllm](https://github.com/joshuamschultz/arcllm)

## License

MIT
