# LLM-as-a-Judge Findings (Depth + Grounding)

## Objective
Evaluate whether **gemini-3-pro-preview** is worth the cost versus **gemini-3-flash-preview** for researcher reports.

## Setup
- Compare run: `reports/compare/20260126_214722` (10 stories)
- Judge model: `google-gla:gemini-3-flash-preview`
- Judge outputs: `eval/judge_20260126_223532.json`, `eval/judge_20260126_223532.md`
- Scoring rubric: 0-5 for **Depth** and **Grounding**, combined as `0.6*depth + 0.4*grounding`

## Findings (Quality)
- Judge winners: **pro 7 / flash 3 / tie 0**
- Average judge scores (combined): **pro 3.47**, **flash 3.38**
- Depth: pro is usually more detailed and structured.
- Grounding compliance regresses with pro:
  - Reports with **zero URLs in Sources**: **pro 8/10**, **flash 4/10**.
  - This is a direct violation of the prompt’s grounding requirement.
- Reliability: pro had **1/10 failures** (502 → empty report).

## Findings (Cost)
Token usage from `reports/compare/20260126_214722/*/*.json`:
- **Flash avg tokens:** input 3,105.4 + output 2,665.9 = **5,771.3**
- **Pro avg tokens:** input 3,457.7 + output 6,963.9 = **10,421.6**
- **Total tokens:** pro is **~1.81x** flash
- **Output tokens:** pro is **~2.61x** flash

## MLE Interpretation
- **Depth improves** with pro, but **grounding compliance worsens**, which is the main product requirement.
- **Cost roughly doubles** while **reliability worsens** (hard failures). This is not acceptable for a default path.
- Current best trade-off: **keep flash as default** and gate pro behind stricter validation.

## Recommended Guardrails
1. **Grounding gate**: reject reports without URL sources or without citations; retry once.
2. **Reliability fallback**: if pro fails, fall back to flash for the same story.
3. **Cost control**: only route to pro when a story is classified as high-impact.

## Decision Snapshot (for now)
- Default to **flash** for production.
- Continue to monitor pro depth wins, but require **grounding compliance** and **failure rate < 2%** before upgrading.
