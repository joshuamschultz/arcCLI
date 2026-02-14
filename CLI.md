# Arc CLI

Unified command-line interface for Arc products.

## Installation

```bash
pip install arccli
```

Or in development:

```bash
pip install -e /path/to/arcllm    # install arcllm first
pip install -e /path/to/arccli    # then arccli
```

## Quick Start

```bash
# See what's available
arc llm providers

# Look at a provider's models and pricing
arc llm provider anthropic

# Make a call
arc llm call anthropic "What is the capital of France?"

# Check your setup
arc llm validate
```

## Commands

### `arc llm version`

Show version information.

```bash
$ arc llm version
arccli : 0.1.0
arcllm : 0.1.0
python : 3.13.9
```

```bash
$ arc llm version --json
{
  "arccli": "0.1.0",
  "arcllm": "0.1.0",
  "python": "3.13.9"
}
```

### `arc llm config`

Show global ArcLLM configuration (defaults, modules, vault).

```bash
$ arc llm config
[defaults]
  provider = anthropic
  temperature = 0.7
  max_tokens = 4096

[modules.retry]
  enabled = False
  max_retries = 3
  backoff_base_seconds = 1.0

...
```

| Flag | Description |
|------|-------------|
| `--module NAME` | Show only a specific module's config |
| `--json` | Output as JSON |

```bash
# Show just telemetry config
arc llm config --module telemetry

# Full config as JSON (for scripting)
arc llm config --json
```

### `arc llm providers`

List all available providers.

```bash
$ arc llm providers
Name             API Format          Default Model
---------------  ------------------  -------------------------------------------------
anthropic        anthropic-messages  claude-sonnet-4-20250514
deepseek         openai-chat         deepseek-chat
openai           openai-chat         gpt-4o
ollama           openai-chat         llama3.2
...
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON array |

### `arc llm provider NAME`

Show provider details: connection settings and all models with metadata.

```bash
$ arc llm provider anthropic
Provider: anthropic

  api_format          : anthropic-messages
  base_url            : https://api.anthropic.com
  api_key_env         : ANTHROPIC_API_KEY
  default_model       : claude-sonnet-4-20250514
  default_temperature : 0.7

Models:
Model                      Context  Max Output  Tools  Vision  Input $/1M  Output $/1M
-------------------------  -------  ----------  -----  ------  ----------  -----------
claude-sonnet-4-20250514   200000   8192        yes    yes     $3.00       $15.00
claude-haiku-4-5-20251001  200000   8192        yes    yes     $0.80       $4.00
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

### `arc llm models`

List all models across all providers in a flat table.

```bash
$ arc llm models
Provider   Model                      Context  Tools  Vision  Input $/1M  Output $/1M
---------  -------------------------  -------  -----  ------  ----------  -----------
anthropic  claude-sonnet-4-20250514   200000   yes    yes     $3.00       $15.00
openai     gpt-4o                     128000   yes    yes     $2.50       $10.00
...
```

| Flag | Description |
|------|-------------|
| `--provider NAME` | Filter by provider |
| `--tools` | Only models that support tool use |
| `--vision` | Only models that support vision |
| `--json` | Output as JSON array |

```bash
# Models that support tools, as JSON
arc llm models --tools --json

# Just Anthropic models
arc llm models --provider anthropic
```

### `arc llm call PROVIDER PROMPT`

Make an LLM call.

```bash
$ arc llm call anthropic "What is 2+2?"
4

$ arc llm call openai "Explain TCP in one sentence" --model gpt-4o
TCP is a reliable, connection-oriented protocol...
```

#### Generation Flags

| Flag | Description |
|------|-------------|
| `--model MODEL` | Override default model |
| `--temperature FLOAT` | Sampling temperature (0.0-2.0) |
| `--max-tokens INT` | Maximum output tokens |
| `--system TEXT` | System message |

#### Module Flags

Toggle modules on/off per call. Default (`None`) uses config.toml settings.

| Flag | Description |
|------|-------------|
| `--retry` / `--no-retry` | Retry with exponential backoff |
| `--fallback` / `--no-fallback` | Fallback to alternate providers |
| `--rate-limit` / `--no-rate-limit` | Token bucket rate limiting |
| `--telemetry` / `--no-telemetry` | Cost and timing tracking |
| `--audit` / `--no-audit` | Audit metadata logging |
| `--security` / `--no-security` | PII redaction + request signing |
| `--otel` / `--no-otel` | OpenTelemetry distributed tracing |

#### Output Flags

| Flag | Description |
|------|-------------|
| `--verbose` | Show usage details (tokens, cost, timing) above response |
| `--json` | Output full LLMResponse as JSON |

```bash
# With telemetry and verbose output
arc llm call anthropic "Hello" --telemetry --verbose
Model: claude-sonnet-4-20250514
Usage: 10 input, 5 output, 15 total tokens
Cost: $0.000105
Stop: end_turn
---
Hello! How can I help you?

# Full JSON response for scripting
arc llm call anthropic "Hello" --json
{
  "content": "Hello! How can I help you?",
  "tool_calls": [],
  "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
  "model": "claude-sonnet-4-20250514",
  "stop_reason": "end_turn",
  "cost_usd": 0.000105
}

# Override everything
arc llm call anthropic "Summarize this" \
  --model claude-haiku-4-5-20251001 \
  --temperature 0.3 \
  --max-tokens 100 \
  --system "Be concise" \
  --retry --no-audit
```

### `arc llm validate`

Validate all configs and check API key availability.

```bash
$ arc llm validate
Global config: OK

Provider         Config  API Key
---------------  ------  ---------------------------
anthropic        OK      MISSING (ANTHROPIC_API_KEY)
openai           OK      OK
ollama           OK      OK
...
```

| Flag | Description |
|------|-------------|
| `--provider NAME` | Validate specific provider only |
| `--json` | Output validation results as JSON |

```bash
# Check just anthropic
arc llm validate --provider anthropic

# JSON for CI/CD pipelines
arc llm validate --json
```

## Global Flag

Every command supports `--json` for machine-readable output. Use it for scripting and CI/CD.

## Help

Every command has built-in help:

```bash
arc --help
arc llm --help
arc llm call --help
```

## Future Commands

The `arc` namespace is extensible. Future products will add:

```bash
arc run ...      # ArcRun commands
arc agent ...    # ArcAgent commands
```
