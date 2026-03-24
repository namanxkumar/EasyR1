# Credit Assignment Bug in Multi-Turn GRPO

## Summary

The current multi-turn rollout only trains on the **last step's (prompt, response) pair** per trajectory. All earlier steps — which the model also generated — receive zero gradient. The model's prior responses are baked into the final prompt as context (memory summaries, prior images) and treated as fixed input, not as model output.

## How it works now

`_build_final_batch` (in `multiturn_tokenizer.py`) constructs one training example per trajectory:

- **Prompt**: `last_prompt` — the prompt from the final step, built by `build_annotate_style_context_from_history`. Contains system prompt, task description, prior observation images, memory summaries from all previous steps, current observation, and instructions. All packed into a single `[system, user]` message pair.
- **Response**: `step_responses[-1]` — only the last step's model output.
- **Reward**: trajectory-level reward placed at the last response token.

GRPO then:
1. Sums `token_level_rewards` across tokens → one scalar per response
2. Groups by UID (N trajectories per dataset item), normalizes within group
3. Broadcasts the same normalized advantage to every response token
4. Computes `pg_loss = -advantage * clipped_ratio` per token

## The problem

- **Steps 0 through N-2 get no gradient.** The model generated responses at every step (reasoning, summaries, navigation actions), but only the final response is in the training batch.
- **Earlier model outputs are laundered into the prompt.** Memory summaries (from the model's `<summary>` tags) and prior observations appear in the user message as fixed context. The optimizer treats them as if a human wrote them.
- **No credit assignment across steps.** If trajectory A succeeds because step 3 navigated to the right room, that decision gets no reinforcement — only the final answer/explore action is trained.

## What the model can/cannot learn

- **Can learn**: Given good final-step context (right room, target visible), produce a correct action.
- **Cannot learn**: How to create good context — earlier navigation decisions that lead to finding the target.

## Fix approach

Include all `(prompt_i, response_i)` pairs from each trajectory in the training batch, each receiving the trajectory reward (or a discounted/shaped version).

Considerations:
- Batch size multiplies by average trajectory length
- GRPO grouping: group all N trajectories' step-i together, or all steps from the same trajectory?
- Reward assignment: same trajectory reward for all steps, or discounted from the end?
- Token budget: earlier prompts are shorter (fewer prior images/summaries), later prompts are longer
