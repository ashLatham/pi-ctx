/**
 * ctx — capture every LLM-bound prompt to disk, and optionally strip
 * thinking/reasoning fields from the context sent to the provider.
 *
 * /ctx                show this help
 * /ctx save           toggle saving on/off (writes <cwd>/ctx/<timestamp>.json)
 * /ctx think          toggle stripping of thinking/reasoning fields from the wire (keeps last 1 by default)
 * /ctx think N        keep last N thinking-bearing messages (default 1, strip all at 0, max KEEP_MAX)
 *
 * Saving: on turn, the LLM-bound payload is serialised to
 * <cwd>/ctx/<timestamp>.json and written atomically (temp + rename) so a reader
 * never sees a half-written file. On `session_start`, when saving is on, the
 * folder is wiped so its size is bounded by the current session's prompts.
 * Off by default; the footer shows live status for both save and think toggles
 * via setStatus.
 *
 * Think-stripping runs on the `before_provider_request` payload — that is the
 * first hook where the LLM-bound wire shape (incl. top-level `reasoning_content`
 * and Anthropic thinking blocks) actually exists. The payload is cloned, pruned,
 * and returned so pi sends the stripped version; saving records exactly what
 * was sent (raw when think is off, stripped when on).
 *
 * "/ctx think N" always enables stripping AND sets keep-last to N — it is a
 * configure, not a toggle. N is clamped to [0, KEEP_MAX]. /ctx save off keeps
 * the existing folder on disk for inspection; the folder is only wiped when
 * saving is toggled back on (or on `session_start` if it was on).
 *
 * Assumption: pi strips reasoning before emitting internal context events, so
 * reasoning content only re-appears on the wire payload. If a future pi version
 * exposes reasoning on an earlier hook, the prune must move there.
 *
 * Default off so an accidental enable doesn't surprise the user with a
 * populated folder.
 */
import { promises as fs } from "node:fs";
import * as path from "node:path";
import { theme, initTheme } from "@earendil-works/pi-coding-agent";

const STATUS_KEY = "ctx";
const FOLDER_NAME = "ctx";

// Save toggle. Off by default so an accidental enable doesn't surprise the user
// with a populated folder.
const saving: { on: boolean } = { on: false };
let thinkEnabled = false;
let thinkKeepLast = 1; // count of most-recent thinking-bearing messages to retain; settable 0..KEEP_MAX
export const KEEP_MAX = 3;

// Sentinel written into stripped fields. A single space is provider-invisible
// (most APIs accept empty/whitespace in string fields) and trivially
// distinguishable from real reasoning content via `isAlreadyPrunedString`.
const PRUNE_MARKER = " ";
const PRUNE_MARKER_MAX_LEN = 2;

// Wire-shape helpers --------------------------------------------------------

interface AssistantMessage {
	role?: string;
	content?: unknown;
	reasoning_content?: unknown;
}

type ContentBlock = Record<string, unknown> & { type?: string };

interface UiLike {
	notify?: (msg: string, level: string) => void;
	setStatus?: (key: string, value: string) => void;
}

interface CtxLike {
	hasUI: boolean;
	ui?: UiLike;
	cwd: string;
}

// Helper: resolve the on-disk output directory for a given cwd.
const outDirFor = (cwd: string) => path.join(cwd, FOLDER_NAME);

// Helper: extract the chat message list from the various wire shapes pi may use
// (top-level `messages` for OpenAI-style, `input` for Anthropic-style).
function getMessageList(payload: unknown): unknown[] | null {
	if (!payload || typeof payload !== "object") return null;
	const p = payload as { messages?: unknown; input?: unknown };
	if (Array.isArray(p.messages)) return p.messages;
	if (Array.isArray(p.input)) return p.input;
	return null;
}

function ts(): string {
	// ISO-ish, filesystem-safe (replaces `:` with `-` because `:` is illegal on
	// Windows), sorts lexically, ms precision. e.g. 2026-06-04T13-05-09-421
	const d = new Date();
	const pad = (n: number, w = 2) => String(n).padStart(w, "0");
	return (
		`${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
		`T${pad(d.getHours())}-${pad(d.getMinutes())}-${pad(d.getSeconds())}-${pad(d.getMilliseconds(), 3)}`
	);
}

async function ensureDir(p: string): Promise<void> {
	await fs.mkdir(p, { recursive: true });
}

// Best-effort wipe+recreate; surfaces failures through `ctx.ui` and returns the
// error (if any) so callers can branch on success/failure.
async function ensureCleanDir(p: string, ctx: CtxLike, label: string): Promise<Error | null> {
	try {
		await wipeDir(p);
		await ensureDir(p);
		return null;
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		if (ctx.hasUI) ctx.ui?.notify?.(`${label}: ${msg}`, "warning");
		return err instanceof Error ? err : new Error(msg);
	}
}

async function wipeDir(p: string): Promise<void> {
	// Best-effort wipe. Missing folder is fine; permission errors propagate.
	await fs.rm(p, { recursive: true, force: true });
}

async function atomicWriteJSON(filePath: string, value: unknown): Promise<void> {
	// Write to a pid-suffixed temp file then rename — POSIX rename is atomic, so
	// a concurrent reader never sees a half-written JSON file.
	const tmp = `${filePath}.${process.pid}.tmp`;
	const data = JSON.stringify(value, null, 2);
	await fs.writeFile(tmp, data, "utf8");
	await fs.rename(tmp, filePath);
}

// Thinking/reasoning stripping ---------------------------------------------

// Rough token estimate: ~4 chars per token. Matches the rule-of-thumb used by
// pi's own UI token counters. Cheap; not accurate.
function estimateTokens(text: string): number {
	return Math.ceil(text.length / 4);
}

// True if `text` looks like the PRUNE_MARKER (whitespace-only, short).
// Used to avoid double-counting tokens when a previous prune pass already
// stripped this field.
function isAlreadyPrunedString(text: string): boolean {
	return text.length > 0 && text.length <= PRUNE_MARKER_MAX_LEN && text.trim() === "";
}

/**
 * Result of a prune pass. Exposed for callers that want to surface metrics
 * (notifications, logs). `tokensFreed` is an estimate.
 */
interface PruneResult {
	messagesScanned: number; // assistant messages with array content
	blocksPruned: number;    // thinking/reasoning fields replaced
	tokensFreed: number;     // estimated chars/4 across pruned fields
}

// Per-block-type pruners. Each receives the block in place and returns the
// estimated tokens freed (0 if the field was already pruned).
type BlockPruner = (block: ContentBlock) => number;

const pruneRedactedThinking: BlockPruner = (b) => {
	// Anthropic encrypted thinking: opaque to us, so convert in place to an
	// empty `text` block and drop encrypted fields.
	const freed = estimateTokens(JSON.stringify(b));
	b.type = "text";
	b.text = PRUNE_MARKER;
	delete b.thinking;
	delete b.data;
	return freed;
};

const pruneThinking: BlockPruner = (b) => {
	// Anthropic standard thinking: { type: "thinking", thinking: "..." }
	const original = typeof b.thinking === "string" ? b.thinking : "";
	return original && !isAlreadyPrunedString(original) ? estimateTokens(original) : 0;
};

const pruneReasoning: BlockPruner = (b) => {
	// OpenAI-style reasoning: { type: "reasoning", text | reasoning: "..." }
	const field =
		typeof b.text === "string"
			? "text"
			: typeof b.reasoning === "string"
				? "reasoning"
				: null;
	if (!field) return 0;
	const original = b[field] as string;
	if (isAlreadyPrunedString(original)) {
		b[field] = PRUNE_MARKER; // normalise: ensure marker is in place even if the
		// field was already short whitespace from a prior pass.
		return 0;
	}
	b[field] = PRUNE_MARKER;
	return estimateTokens(original);
};

// Dispatch table for block-type → pruner. Adding a new provider's thinking
// variant = add one entry + one branch in the "is thinking-bearing" check.
const BLOCK_PRUNERS: Record<string, BlockPruner> = {
	redacted_thinking: pruneRedactedThinking,
	thinking: pruneThinking,
	reasoning: pruneReasoning,
};

const THINKING_BLOCK_TYPES = new Set(Object.keys(BLOCK_PRUNERS));

// True if the message carries thinking/reasoning content we may need to strip.
// Matches either a top-level `reasoning_content` string (DeepSeek R1 / some
// OpenAI-compatible providers) OR any content block in BLOCK_PRUNERS.
function isThinkingBearing(msg: AssistantMessage): boolean {
	if (typeof msg.reasoning_content === "string") return true;
	if (!Array.isArray(msg.content)) return false;
	return (msg.content as ContentBlock[]).some(
		(b) => b && typeof b === "object" && typeof b.type === "string" && THINKING_BLOCK_TYPES.has(b.type),
	);
}

/**
 * Strips thinking/reasoning content out of assistant message content in place.
 *
 * Handles DeepSeek R1-style top-level `reasoning_content` strings plus Anthropic
 * `{type:"thinking"}` / `{type:"redacted_thinking"}` / OpenAI-style
 * `{type:"reasoning"}` blocks. Retains the last `thinkKeepLast` thinking-bearing
 * messages intact; 0 strips all. Mutates `messages` in place — providers care
 * about block indices, so we replace reasoning fields with `PRUNE_MARKER`
 * rather than dropping blocks outright.
 */
function pruneThinkingBlocks(messages: unknown[]): PruneResult {
	if (!Array.isArray(messages)) return { messagesScanned: 0, blocksPruned: 0, tokensFreed: 0 };

	let messagesScanned = 0;
	let blocksPruned = 0;
	let tokensFreed = 0;

	// First pass: collect indices of assistant messages that carry thinking so
	// we know which slice to keep.
	const thinkMsgIndices: number[] = [];
	for (let i = 0; i < messages.length; i++) {
		const msg = messages[i] as AssistantMessage;
		if (!msg || msg.role !== "assistant") continue;
		if (isThinkingBearing(msg)) thinkMsgIndices.push(i);
	}

	const keepCount = Math.max(0, Math.min(thinkKeepLast, KEEP_MAX));
	// Special-case keepCount === 0: `slice(-0) === slice(0) === whole array`,
	// which would keep everything. Empty set = keep nothing.
	const toKeep = keepCount === 0 ? new Set<number>() : new Set(thinkMsgIndices.slice(-keepCount));

	// Second pass: mutate in place.
	for (let mi = 0; mi < messages.length; mi++) {
		const msg = messages[mi] as AssistantMessage;
		if (!msg || msg.role !== "assistant") continue;
		if (toKeep.has(mi)) continue;

		// DeepSeek R1 / some OpenAI-compatible: top-level "reasoning_content" string
		if (typeof msg.reasoning_content === "string" && !isAlreadyPrunedString(msg.reasoning_content)) {
			tokensFreed += estimateTokens(msg.reasoning_content);
			msg.reasoning_content = PRUNE_MARKER;
			blocksPruned++;
		}

		const content = msg.content;
		if (typeof content === "string") continue;
		if (!Array.isArray(content)) continue;

		messagesScanned++;
		for (const block of content) {
			if (!block || typeof block !== "object") continue;
			const blockType = (block as ContentBlock).type;
			if (typeof blockType !== "string") continue;
			const pruner = BLOCK_PRUNERS[blockType];
			if (!pruner) continue;
			tokensFreed += pruner(block as ContentBlock);
			blocksPruned++;
		}
	}

	return { messagesScanned, blocksPruned, tokensFreed };
}

// Footer status indicator --------------------------------------------------

function dimColor(text: string): string {
	// Match the footer's stats line styling exactly. The footer wraps stats
	// text in theme.fg("dim", ...) — do the same so colors derive from the
	// active theme and stay in sync if the theme changes.
	try {
		const result = theme.fg("dim", text);
		return typeof result === "string" ? result : text;
	} catch {
		// Theme not initialized yet — fall back to ANSI bright black (90).
		// Intentionally always emits escapes: theme.fg() should normally have
		// run; this fallback only fires during early startup. If NO_COLOR is
		// set, callers can wrap the result with their own stripper.
		return `\x1b[90m${text}\x1b[39m`;
	}
}

function statusText(): string {
	// Build with per-segment coloring:
	//   - each label is white when its toggle is on, dim when off
	//   - "|" is dim if EITHER neighbour is dim, white only if BOTH are white
	//   - "ctx(", ")" are dim when ALL labels are dim, white otherwise
	// Examples:
	//   save=on,  think=on  → "ctx(save|think)"       (all white)
	//   save=on,  think=off → "ctx(save|think)"       (save white, |+think dim)
	//   save=off, think=off → "ctx(save|think)"       (all dim)
	const labels: [string, boolean][] = [
		["save", saving.on],
		["think", thinkEnabled],
	];
	const anyOn = labels.some(([, on]) => on);
	const wrapDim = !anyOn;

	let result = wrapDim ? dimColor("ctx(") : "ctx(";
	for (let i = 0; i < labels.length; i++) {
		const [label, on] = labels[i];
		result += on ? label : dimColor(label);
		if (i < labels.length - 1) {
			const [, nextOn] = labels[i + 1];
			const sepDim = !on || !nextOn;
			result += sepDim ? dimColor("|") : "|";
		}
	}
	result += wrapDim ? dimColor(")") : ")";
	return result;
}

// Extension entry point ----------------------------------------------------

export default function ctx(pi: ExtensionAPI): void {
	// Defensive: host usually calls initTheme() at startup; do it here too so the
	// extension works even when loaded in non-interactive modes. (If pi exposes
	// a dedicated init lifecycle hook in the future, move this there.)
	try {
		initTheme();
	} catch {
		/* already initialized or unavailable */
	}

	let outDir = ""; // resolved lazily from ctx.cwd so per-session layout is respected

	const refreshStatus = (ctx: CtxLike) => {
		if (ctx.hasUI) ctx.ui?.setStatus?.(STATUS_KEY, statusText());
	};

	pi.on("session_start", async (_event, ctx) => {
		outDir = outDirFor(ctx.cwd);
		// Only wipe when the user has actually turned saving on. Off = leave prior
		// files alone so a later enable doesn't surprise the user with lost data.
		if (saving.on) {
			await ensureCleanDir(outDir, ctx, "ctx: wipe failed");
		}
		refreshStatus(ctx);
	});

	pi.on("before_provider_request", async (event, ctx) => {
		const wirePayload: unknown = (event as { payload: unknown }).payload;
		const payloadAny = wirePayload as Record<string, unknown>;

		// Build the exact payload pi will send. If think is on, clone + prune
		// reasoning blocks and return the stripped clone. Otherwise pass through.
		let sentPayload: unknown = wirePayload;
		if (thinkEnabled) {
			const srcList = getMessageList(payloadAny);
			if (srcList) {
				const cloneResult = cloneAndPrune(payloadAny, ctx);
				sentPayload = cloneResult ?? wirePayload;
			}
			// No `messages` or `input` field: don't prune, but still save the raw
			// payload (handled below) so disk reflects exactly what was sent.
		}

		if (saving.on) {
			if (!outDir) outDir = outDirFor(ctx.cwd);
			await recordPayload(sentPayload, outDir, ctx);
		}

		return sentPayload;
	});

	pi.registerCommand("ctx", {
		description: "Toggle ctx save/think; captures LLM prompts to <cwd>/ctx and optionally strips thinking",
		handler: async (args, ctx) => {
			const arg = (args ?? "").trim().toLowerCase();

			if (arg === "") {
				if (ctx.hasUI) {
					ctx.ui?.notify?.(
						`/ctx save — toggle saving on/off · /ctx think — toggle strip · /ctx think N — keep last N (0–${KEEP_MAX})`,
						"info",
					);
				}
				return;
			}

			if (arg === "save") return handleSaveToggle(ctx);
			if (arg === "think") return handleThinkToggle(ctx);

			const m = /^think\s+(\d+)$/.exec(arg);
			if (m) return handleThinkSet(Number(m[1]), ctx);

			if (ctx.hasUI) {
				ctx.ui?.notify?.(
					`Unknown parameter. Use: /ctx save — toggle saving, /ctx think — toggle strip, /ctx think N — keep last N (0–${KEEP_MAX})`,
					"error",
				);
			}
		},
	});
}

// Command handlers ---------------------------------------------------------

async function handleSaveToggle(ctx: CtxLike): Promise<void> {
	const next = !saving.on;
	if (next === saving.on) {
		if (ctx.hasUI) ctx.ui?.notify?.(`ctx save already ${next ? "on" : "off"}`, "info");
		return;
	}
	saving.on = next;
	if (next) {
		const dir = outDirFor(ctx.cwd);
		// Wipe-and-recreate on enable so old session data doesn't bleed into the
		// new one. Roll back the toggle if this fails so the footer doesn't lie
		// about a directory we couldn't prepare.
		const err = await ensureCleanDir(dir, ctx, "ctx save: wipe failed");
		if (err) {
			saving.on = false;
			refreshStatusFor(ctx);
			return;
		}
		if (ctx.hasUI) ctx.ui?.notify?.(`ctx save: on (writing to ${dir})`, "info");
	} else {
		if (ctx.hasUI) ctx.ui?.notify?.(`ctx save: off (kept ${outDirFor(ctx.cwd)}; will wipe on next enable)`, "info");
	}
	refreshStatusFor(ctx);
}

function handleThinkToggle(ctx: CtxLike): void {
	thinkEnabled = !thinkEnabled;
	refreshStatusFor(ctx);
	if (ctx.hasUI) {
		ctx.ui?.notify?.(
			thinkEnabled
				? `ctx think ON — stripping reasoning, keeping last ${thinkKeepLast}`
				: "ctx think OFF — thinking/reasoning included",
			"info",
		);
	}
}

function handleThinkSet(n: number, ctx: CtxLike): void {
	const clamped = Math.max(0, Math.min(n, KEEP_MAX));
	// Numeric form is a configure, not a toggle: always enables stripping and
	// sets the window. Otherwise a user typing `/ctx think 0` while think was
	// off would see "0" silently retained but never apply.
	thinkKeepLast = clamped;
	thinkEnabled = true;
	refreshStatusFor(ctx);
	if (ctx.hasUI) {
		ctx.ui?.notify?.(
			`ctx think ON — keeping last ${clamped} (${clamped === 0 ? "stripping all" : `last ${clamped}`}), stripping enabled`,
			"info",
		);
	}
}

// Wire-payload helpers -----------------------------------------------------

function refreshStatusFor(ctx: CtxLike): void {
	if (ctx.hasUI) ctx.ui?.setStatus?.(STATUS_KEY, statusText());
}

// Clone `payload` and prune thinking blocks on the clone. Returns null on
// clone failure (caller should send the un-stripped wire payload). Notifies
// the user on failure when UI is available.
function cloneAndPrune(payload: Record<string, unknown>, ctx: CtxLike): Record<string, unknown> | null {
	let clone: Record<string, unknown>;
	try {
		clone = structuredClone(payload);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		if (ctx.hasUI) ctx.ui?.notify?.(`ctx think: clone failed, sending unstripped: ${msg}`, "warning");
		return null;
	}
	const list = getMessageList(clone);
	if (list) pruneThinkingBlocks(list);
	return clone;
}

async function recordPayload(payload: unknown, outDir: string, ctx: CtxLike): Promise<void> {
	try {
		await ensureDir(outDir);
		await atomicWriteJSON(path.join(outDir, `${ts()}.json`), payload);
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		if (ctx.hasUI) ctx.ui?.notify?.(`ctx: write failed: ${msg}`, "warning");
	}
}
