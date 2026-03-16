---
name: fact-check
description: Systematically verify factual claims in documents using multi-LLM triage and source-grounded verification via MCP servers and web search. Use when user says "fact-check", "fact check", "verify accuracy", "check this document", "accuracy review".
allowed-tools: Bash, Read, Write, Edit, Agent, AskUserQuestion, WebFetch, WebSearch, Grep, Glob, mcp__microsoft_docs_mcp__microsoft_docs_search, mcp__microsoft_docs_mcp__microsoft_docs_fetch, mcp__microsoft_docs_mcp__microsoft_code_sample_search, mcp__plugin_context7_context7__resolve-library-id, mcp__plugin_context7_context7__query-docs
---

# Fact-Check: Document Accuracy Verification

## Overview

Verify factual claims in research, governance, and reference documents using a hybrid approach:
1. **Multi-LLM triage** — Send claims to multiple models, flag uncertain/suspect ones
2. **Source-grounded verification** — Deep-verify flagged claims against authoritative sources (MCP servers, web search, official docs)
3. **Report & fix** — Structured accuracy report, then optionally apply corrections

## Scripts Location

All scripts are in the skill's `scripts/` directory (relative to the plugin root):
```
skills/fact-check/scripts/
├── verify.py      # Main orchestrator + CLI
├── models.py      # LiteLLM calls, parallel execution, cost tracking
├── prompts.py     # Claim extraction + verification prompt templates
├── providers.py   # Provider config + cost tracking tables
└── sources.py     # MCP detection + source resolution + registry
```

**Requires:** `pip install litellm` (or `pip3 install litellm` / `python3 -m pip install litellm` if `pip` is not found)

## Workflow

Follow these steps in order. Each step has a user checkpoint — do NOT skip ahead without confirmation.

### Step 0: PREFLIGHT CHECK

Before starting, check what LLM providers are available:
```bash
cd ${CLAUDE_PLUGIN_ROOT}/skills/fact-check/scripts && python3 verify.py providers
```

**If no external providers are found** (no API keys set, no CLI tools installed), warn the user:

> **Warning: No external LLM providers detected.** Multi-LLM triage works best with 2-3 independent models to cross-check claims. Without external models, I'll use my own knowledge for triage — this still works but you lose the independent verification that catches blind spots.
>
> **Recommended setup (pick one):**
> - Set `OPENROUTER_API_KEY` for access to multiple providers with a single key ([openrouter.ai](https://openrouter.ai))
> - Or set API keys for 2+ providers (e.g. `OPENAI_API_KEY`, `GEMINI_API_KEY`)
> - Or install Codex CLI (`npm install -g @openai/codex`) or Gemini CLI (`npm install -g @google/gemini-cli`)

Then ask: "Continue anyway with single-model triage, or set up providers first?" Proceed if the user says to continue.

**IMPORTANT — no simulated triage:** If no external LLMs are available, do NOT launch subagents that role-play or simulate different LLM perspectives. That is not real independent verification — it's just you (Claude) with extra steps. Instead, skip the multi-model triage entirely and flag ALL claims for deep source-grounded verification in Step 5. Be transparent with the user that triage was skipped because no external models were available.

### Step 1: INGEST

1. Ask the user for the document to verify (file path, or they may have already provided it)
2. Read the document using the Read tool
3. Run domain detection:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT}/skills/fact-check/scripts && python3 -c "
   from sources import detect_document_domains, find_relevant_mcps, load_registry
   content = open('<FILE_PATH>').read()
   domains = detect_document_domains(content)
   print('Detected domains:', domains)
   "
   ```
4. Check which MCP servers are available for verification
5. Report to user: document length, detected domains, available MCPs, recommended MCPs

### Step 2: EXTRACT

1. Read the document content
2. Using the prompt templates from `prompts.py`, extract all verifiable factual claims
3. For each claim, capture: id, text (exact quote), category, section

**Claim categories:** pricing, capability, date, licensing, compliance, architecture, status, number

**What to extract:**
- Pricing figures ($X/user/month, included in Y license)
- Dates (GA dates, announcement dates, availability windows)
- Capability descriptions (tool X does Y, supports Z)
- Status claims (preview, GA, deprecated)
- Numbers (320+ templates, 30+ providers)
- Architecture claims (uses X for Y, integrates with Z)
- Compliance claims (certified for X, meets Y standard)
- Licensing details (included in E5, requires add-on)

**What NOT to extract:**
- Opinions, recommendations, strategy
- Internal process statements
- Vague qualitative claims ("comprehensive", "robust")

### Step 3: REVIEW (User Checkpoint 1)

1. Present the extracted claim list to the user in a table:
   ```
   | # | Category | Claim | Section |
   |---|----------|-------|---------|
   ```
2. Ask: "Review this claim list. Remove any you don't need verified, add any I missed."
3. Wait for user confirmation before proceeding

### Step 4: TRIAGE

1. Check available LLM providers:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT}/skills/fact-check/scripts && python3 verify.py providers
   ```

2. Select 2-3 models from different providers. Pick from whatever is available — run `python3 verify.py providers` to see current models. Aim for diversity across providers, e.g.:
   - One OpenAI model
   - One Google Gemini model
   - One xAI or other provider model

   If user has a saved profile, use that instead:
   ```bash
   python3 verify.py profiles list
   ```

3. Run triage — send ALL claims to selected models in parallel:
   ```bash
   cd ${CLAUDE_PLUGIN_ROOT}/skills/fact-check/scripts && python3 verify.py triage claims.json --models <model1>,<model2>,<model3> --document <FILE_PATH>
   ```

   **If the CLI triage is not practical** (e.g., claims aren't in a file yet), you can orchestrate triage directly:
   - Use the Agent tool to launch parallel agents, one per model
   - Each agent gets the triage system prompt from `prompts.py` and the claim list
   - Aggregate results: CONFIDENT from all models = model-verified; any UNCERTAIN/SUSPECT = flagged

4. **Triage aggregation rules:**
   - All models say CONFIDENT → claim is **model-verified** (skip deep verification)
   - Any model says UNCERTAIN → claim **flagged** for deep verification
   - Any model says SUSPECT → claim **flagged HIGH PRIORITY** for deep verification

5. Present triage results to user (User Checkpoint 2):
   ```
   Model-verified (all CONFIDENT): X claims
   Flagged for deep verification: Y claims

   Flagged claims:
   | # | Category | Claim | Flagged by | Reason |
   ```
6. Ask: "Proceed with deep verification of these Y claims?"

### Step 5: VERIFY

For each flagged claim, perform source-grounded verification:

1. **Build source plan** — match each claim to the best available source:
   - Microsoft claims → `microsoft_docs_search` / `microsoft_docs_fetch` MCP tools
   - Library/framework claims → `context7` MCP tools
   - GitHub/open-source claims → `deepwiki` MCP tools (if connected)
   - All others → WebSearch / WebFetch
   - Model knowledge as last resort

2. **Execute verification in parallel** using the Agent tool:
   - Launch parallel agents grouped by source (e.g., all Microsoft claims in one batch)
   - Each agent searches the authoritative source, then assesses the claim against what it finds
   - Use the verification prompt from `prompts.py` (`VERIFY_SYSTEM` + `VERIFY_USER`)

3. **For each claim, capture:**
   - Verdict: CONFIRMED / NUANCED / INCORRECT / OUTDATED / UNCONFIRMED
   - Source: URL or MCP tool used
   - Quote: Relevant excerpt from source
   - Explanation: Why this verdict
   - Suggested fix: Replacement text (for NUANCED/INCORRECT/OUTDATED)

4. **Verdict definitions:**
   - **CONFIRMED** — Source directly supports the claim as stated
   - **NUANCED** — Substantively correct but imprecise, overstated, or missing qualifications
   - **INCORRECT** — Source contradicts the claim
   - **OUTDATED** — Was once true but source shows it has changed
   - **UNCONFIRMED** — Source neither confirms nor denies (insufficient evidence)

### Step 6: REPORT

1. Generate the accuracy report using the template from `prompts.py`
2. Save report alongside the source document:
   - Report path: `{source_dir}/{source_name}-fact-check-{YYYY-MM-DD}.md`
3. Display the report to the user

**Report structure:**
- Summary (total claims, verdict counts, cost)
- HIGH PRIORITY: Incorrect / Outdated claims with suggested fixes
- MEDIUM PRIORITY: Nuanced claims with precision improvements
- LOW PRIORITY: Unconfirmed claims
- VERIFIED: Confirmed + model-verified claims (collapsed)
- MCP RECOMMENDATIONS: Missing MCPs that would improve coverage

### Step 7: FIX (User Checkpoint 3)

1. Ask: "Apply fixes to the source document?"
2. If yes, walk through each HIGH and MEDIUM finding:
   - Show the original text in the document
   - Show the suggested replacement
   - Ask: "Apply this fix? (yes/skip/edit)"
3. For each approved fix, use the Edit tool to apply the change
4. After all fixes, show summary of changes made

## Key Principles

- **Evidence before assertions** — every verdict must cite a source
- **Precision matters** — "320+" vs "360+" is the kind of error that erodes credibility
- **User checkpoints** — three gates before money is spent or documents changed
- **MCP-first** — prefer authoritative MCP sources over web search
- **Cost transparency** — show token usage and cost at every stage
- **Conservative triage** — when in doubt, flag for deep verification

## CLI Reference

```bash
# Navigate to scripts directory first
cd ${CLAUDE_PLUGIN_ROOT}/skills/fact-check/scripts

# Full pipeline analysis
python3 verify.py check <file>

# Extract claims only
python3 verify.py extract <file>

# Triage extracted claims
python3 verify.py triage claims.json --models <model1>,<model2> --document <file>

# List available LLM providers
python3 verify.py providers

# List MCP source registry
python3 verify.py sources

# Manage profiles
python3 verify.py profiles list
python3 verify.py profiles save --name gov-check --models <model1>,<model2>

# Manage MCP registry
python3 verify.py registry list
python3 verify.py registry add --name my-mcp --domains "cloud,infrastructure" --description "My custom MCP"
```

## Orchestration Notes for Claude

When executing this skill, you (Claude) are the orchestrator. The Python scripts handle:
- Provider detection and credential validation
- Cost tracking across models
- Parallel model invocation for triage
- MCP registry management

But YOU handle:
- Reading documents (Read tool)
- Extracting claims (your own analysis using the prompt templates as guidance)
- Source verification (using MCP tools, WebSearch, WebFetch directly)
- Report generation (using the template format)
- Fix application (using Edit tool)
- User interaction at checkpoints (present findings, ask for confirmation)

The scripts are helpers, not the whole pipeline. You drive the workflow.

## MCP Source Registry

The skill ships with a default registry of known MCPs. Users can extend it:

| MCP | Domains | Description |
|-----|---------|-------------|
| microsoft_docs_mcp | Microsoft, Azure, M365, Entra, etc. | Official Microsoft Learn docs |
| context7 | Libraries, frameworks, SDKs | Up-to-date library documentation |
| deepwiki | GitHub, open-source repos | Repository docs and wikis |
| atlassian | Jira, Confluence | Atlassian products |

At runtime:
1. Check what MCPs are connected
2. Match document domains to registry
3. Use connected MCPs for verification
4. Recommend unconnected MCPs that would help
