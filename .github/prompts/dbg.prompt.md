---
name: dbg
description: "Alias for inline adversarial debugging. Use when you want /dbg followed by a failure description."
argument-hint: "Type the failure, error, failing command, test, regression, or anchor file after /dbg."
agent: "Adversarial Debug Loop"
---
Use the Adversarial Debug Loop agent for the debugging task provided after this slash command.

Treat the user's trailing prompt text as the task description.

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