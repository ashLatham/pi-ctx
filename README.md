# pi-ctx

Capture every LLM-bound prompt to disk, and optionally strip thinking/reasoning fields from the context sent to the provider. Designed for debugging context-window contents, replaying prompts offline, and shrinking wire payloads when chain-of-thought bloat is costing you tokens and context.

## Features

**On-disk prompt capture** — When enabled (default off), on every turn the exact payload pi sends to the provider is serialised to `<cwd>/ctx/<timestamp>.json`. Writes are atomic (temp file + rename) so a concurrent reader never sees a half-written file. On `session_start`, when saving is on, the folder is wiped so its size is bounded by the current session's prompts. Off by default — the folder is never created until you opt in.

**Thinking/reasoning stripping** — Strips chain-of-thought out of the wire payload *before* pi sends it. Handles DeepSeek R1-style top-level `reasoning_content` strings, Anthropic `{type:"thinking"}` and `{type:"redacted_thinking"}` blocks, and OpenAI-style `{type:"reasoning"}` blocks. Retains the last N thinking-bearing messages intact (configurable, see Usage). Default off.

**Footer status indicator** — Live `ctx(save|think)` indicator in the pi footer:
- label is white when the toggle is on, dim when off
- `|` is dim if either neighbour is dim, white only if both are white
- `ctx(` / `)` are dim when all labels are dim, white otherwise

**Idempotent and safe** — Saving and thinking are independent toggles. Disabling saving leaves the existing folder on disk for inspection. Stripping is a no-op when the payload carries no reasoning fields. Clone failures fall back to the un-stripped payload with a UI warning rather than failing the turn.

## Installation

Install from npm:

```bash
pi install npm:pi-ctx
```

Install into the current project only:

```bash
pi install npm:pi-ctx -l
```

Or install from GitHub:

```bash
pi install git:github.com/ashLatham/pi-ctx
```

Try it without permanently installing:

```bash
pi -e npm:pi-ctx
```


## Usage

All configuration is via the `/ctx` slash command. No config file.

```
/ctx                       show usage
/ctx save                  toggle saving on/off
/ctx think                 toggle thinking/reasoning stripping on/off
/ctx think N               keep last N thinking-bearing messages (0..3, default 1)
```

Examples:
- `/ctx save` then send a prompt — find the captured JSON in `<cwd>/ctx/`.
- `/ctx think` — strip all but the most recent reasoning block.
- `/ctx think 0` — strip all reasoning. Note: this is a *configure*, not a toggle; it always enables stripping.
- `/ctx think 3` — keep reasoning on the last three assistant turns.

### Notification levels

- `info` — toggle state changes (`ctx save: on (writing to <dir>)`, `ctx think ON — keeping last 1`, etc.)
- `warning` — non-fatal failures: wipe failed, clone failed (sends un-stripped), write failed
- `error` — unknown `/ctx` argument

### What gets stripped

A message is "thinking-bearing" if it has any of:
- a top-level `reasoning_content` string field (DeepSeek R1 / some OpenAI-compatible providers)
- a content block with `type` of `thinking`, `redacted_thinking`, or `reasoning`

When a message is stripped, the reasoning fields are replaced with a single-space sentinel rather than dropped entirely — providers care about block indices and message shape, so removing fields would risk breaking the request.

### Saved payload fidelity

The JSON written to `<cwd>/ctx/` is *exactly* what pi sent to the provider:
- think off → raw wire payload
- think on → stripped payload (after pruning)


## Links

- npm: https://www.npmjs.com/package/pi-ctx
- GitHub: https://github.com/ashLatham/pi-ctx

## License
MIT
