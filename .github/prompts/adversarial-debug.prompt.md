---
name: adversarial-debug
description: "Force the Adversarial Debug Loop agent for a debugging task. Use when you want the adversarial debugger instead of the default agent."
argument-hint: "Describe the failure, error, failing command, test, regression, or anchor file."
agent: "Adversarial Debug Loop"
---
Use the Adversarial Debug Loop agent for this debugging task:

${input:debug_task:Describe the failure, error, failing command, test, or anchor file.}

Requirements:
- Start from the most concrete anchor available.
- State one falsifiable local hypothesis and one cheap discriminating check.
- Invoke Hypothesis Challenger before the first edit.
- Validate immediately after the first substantive edit.
- Invoke Regression Skeptic after each substantive validation result.
- Continue until focused validation passes or there is a concrete blocker with evidence.

Return:
- Anchor
- Current hypothesis
- Adversarial challenge
- Chosen action
- Validation
- Result or blocker