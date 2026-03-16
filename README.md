# fact-checker

A Claude Code plugin that systematically verifies factual claims in documents using multi-LLM triage and source-grounded verification.

Built primarily for verifying **Microsoft and Azure technical documentation** (pricing, licensing, capabilities, dates, compliance), but works with most technical content — library docs, cloud platform references, governance documents, and more.

## How It Works

A 7-step pipeline orchestrated by Claude:

1. **Ingest** — Read the document and detect knowledge domains
2. **Extract** — Identify verifiable factual claims (pricing, dates, capabilities, etc.)
3. **Review** — User checkpoint to validate the claim list
4. **Triage** — Send claims to 2-3 LLMs in parallel; flag anything uncertain or suspect
5. **Verify** — Deep-verify flagged claims against authoritative sources (MCP servers, official docs, web search)
6. **Report** — Structured accuracy report with priorities and suggested fixes
7. **Fix** — Optionally apply corrections to the source document

Claims that all models agree on skip deep verification. Claims where any model is uncertain get source-grounded verification against authoritative documentation.

## Installation

```
/plugin marketplace add machug/fact-checker
/plugin install fact-checker
```

### Requirements

- Python 3.10+
- `pip install litellm`
- API keys for at least 2 LLM providers (set as environment variables)

### Supported Providers

| Provider | Env Variable |
|----------|-------------|
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| xAI | `XAI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Deepseek | `DEEPSEEK_API_KEY` |

**Quickest way to get started:** [OpenRouter](https://openrouter.ai/) gives you access to models from OpenAI, Google, Anthropic, xAI, and others with a single API key — no need to create separate accounts with each provider.

Available models and pricing (USD) are discovered dynamically from [LiteLLM's model registry](https://github.com/BerriAI/litellm) — no hardcoded model lists to go stale. Run `python3 verify.py providers` to see what's available with your current keys.

The plugin auto-detects what's available — it checks for API keys in your environment **and** installed CLI tools. If you have Codex CLI (`@openai/codex`) or Gemini CLI (`@google/gemini-cli`) installed, those work too without needing an API key (they use your existing subscription/account).

## Usage

In Claude Code, just say:

```
fact-check this document: path/to/document.md
```

Or any variation like "verify accuracy of...", "check this document", etc.

Claude walks you through each step with checkpoints before spending tokens or modifying files.

### CLI Tools

The plugin also includes CLI helpers you can run directly:

```bash
# List available providers
python3 verify.py providers

# Extract claims from a document
python3 verify.py extract <file>

# Run triage with specific models (use any models from `verify.py providers`)
python3 verify.py triage claims.json --models <model1>,<model2>,<model3> --document <file>

# Manage model profiles
python3 verify.py profiles list
python3 verify.py profiles save --name my-profile --models <model1>,<model2>

# View/extend the MCP source registry
python3 verify.py registry list
python3 verify.py registry add --name my-mcp --domains "cloud,infra" --description "My custom MCP"
```

## MCP Integration

The plugin is MCP-aware — it prefers authoritative sources over web search. Built-in MCP mappings:

| MCP Server | Domains | Use |
|------------|---------|-----|
| microsoft_docs_mcp | Microsoft, Azure, M365, Entra | Official Microsoft Learn documentation |
| context7 | Libraries, frameworks, SDKs | Up-to-date library/framework docs |
| deepwiki | GitHub, open-source | Repository documentation and wikis |

You can extend the registry with your own MCP servers for custom domain coverage.

## Verdicts

Each verified claim gets one of:

- **CONFIRMED** — Source directly supports the claim
- **NUANCED** — Correct but imprecise, overstated, or missing qualifications
- **INCORRECT** — Source contradicts the claim
- **OUTDATED** — Was true but has since changed
- **UNCONFIRMED** — Insufficient evidence either way

## Cost Tracking

Token usage and costs (USD) are tracked across all LLM calls and reported at every stage, so you always know what you're spending.

## License

MIT
