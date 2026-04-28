<script lang="ts">
	import { Search, Cloud, Code, Image, Globe } from '@lucide/svelte';
	import SyntaxHighlightedCode from '$lib/components/app/content/SyntaxHighlightedCode.svelte';
	import type { ApiToolArtifactPayload } from '$lib/types';

	interface Props {
		tool: string;
		status: string;
		query?: string;
		expanded?: boolean;
		onToggleExpand?: () => void;
		artifacts?: ApiToolArtifactPayload[];
		argStream?: { field: string; text: string; complete: boolean };
	}

	let { tool, status, query, expanded = false, onToggleExpand, artifacts, argStream }: Props = $props();

	const iconMap: Record<string, typeof Search> = {
		web_search: Search,
		weather: Cloud,
		code_exec: Code,
		image_gen: Image,
		url_fetch: Globe
	};

	const labelMap: Record<string, string> = {
		web_search: 'Searching',
		weather: 'Weather',
		code_exec: 'Running code',
		image_gen: 'Generating image',
		url_fetch: 'Fetching'
	};

	const writingLabelMap: Record<string, string> = {
		code_exec: 'Writing code',
		web_search: 'Forming query',
		image_gen: 'Composing prompt',
		url_fetch: 'Choosing URL'
	};

	// Merged-orchestration namespaces server-side tools as `__server.<name>`
	// (e.g. `__server.code_exec`). Strip the prefix for icon/label lookup so
	// the chip displays the friendly form regardless of session mode.
	let bareTool = $derived(tool.startsWith('__server.') ? tool.slice('__server.'.length) : tool);
	let IconComponent = $derived(iconMap[bareTool] ?? Globe);
	let label = $derived(labelMap[bareTool] ?? bareTool);
	let isWriting = $derived(
		!!argStream && !argStream.complete && status !== 'completed' && status !== 'failed'
	);
	let writingLabel = $derived(writingLabelMap[bareTool] ?? 'Writing');
	let displayLabel = $derived(isWriting ? writingLabel : label);
	let statusColor = $derived(
		status === 'completed'
			? 'var(--green-500, #22c55e)'
			: status === 'failed'
				? 'var(--red-500, #ef4444)'
				: 'var(--amber-500, #f59e0b)'
	);
	let isExecuting = $derived(status === 'executing' || isWriting);
	let isClickable = $derived((!!query || !!argStream) && !!onToggleExpand);
	let imageArtifacts = $derived(
		(artifacts ?? []).filter((a) => a.kind === 'image')
	);

	// code_exec gets a detached, syntax-highlighted block (sibling of the
	// chip) instead of the bespoke monospace span nested inside the pill.
	// The pill stays small; the code lives below it as its own block — so
	// the chip identity isn't crushed by the body of the code.
	let isCodeExec = $derived(bareTool === 'code_exec');
	let showDetachedCode = $derived(isCodeExec && !!argStream && expanded);
	let inlineArgStream = $derived(!!argStream && !query && !isCodeExec);

	let codeContainer: HTMLDivElement | undefined = $state();
	// Auto-scroll is "sticky": engaged while the user's view is at (or
	// near) the bottom of the code wrapper. If they scroll up to read an
	// earlier line, we stop yanking them back; when they scroll back down
	// to the tail, auto-scroll re-engages. Default is true so the first
	// few streamed deltas pin to the bottom out of the gate.
	let stickToBottom = $state(true);
	const STICK_TOLERANCE_PX = 8;

	// Streaming-only auto-scroll: while the model is writing the code, pin
	// the visible window to the latest line so the user always sees what
	// just arrived. Once the stream completes, reset to the top — the
	// reader expects to start from line 1 when re-opening the panel.
	$effect(() => {
		if (!argStream || !codeContainer) return;
		const wrapper = codeContainer.querySelector('.code-preview-wrapper');
		if (!(wrapper instanceof HTMLElement)) return;

		if (argStream.complete) {
			wrapper.scrollTop = 0;
			// Reset for any subsequent re-streaming on the same chip
			// (rare, but keeps the state machine clean).
			stickToBottom = true;
			return;
		}

		// Read text inside the effect to register a reactive dependency
		// on every fragment update. The actual scroll runs in rAF so the
		// child SyntaxHighlightedCode has finished patching the DOM with
		// the new highlighted HTML — without that, scrollHeight reflects
		// the previous frame and we'd land one line short.
		const text = argStream.text;
		if (!text) return;
		requestAnimationFrame(() => {
			// Read stickToBottom inside rAF so a user scroll that lands
			// between effect-fire and frame-paint still wins. Reading
			// here doesn't add a reactive dep (we're outside the effect
			// closure scope by the time rAF runs) — that's intentional;
			// the effect should re-fire on text/complete changes only.
			if (!stickToBottom) return;
			wrapper.scrollTop = wrapper.scrollHeight;
		});
	});

	// Manual-scroll detection: flip stickToBottom off when the user pulls
	// the viewport away from the tail, on again when they return to it.
	// Programmatic scrolls (the rAF above) also fire scroll events, but
	// they always land *at* the bottom so they leave stickToBottom=true.
	$effect(() => {
		if (!codeContainer) return;
		const wrapper = codeContainer.querySelector('.code-preview-wrapper');
		if (!(wrapper instanceof HTMLElement)) return;

		const onScroll = () => {
			const distanceFromBottom =
				wrapper.scrollHeight - (wrapper.scrollTop + wrapper.clientHeight);
			stickToBottom = distanceFromBottom <= STICK_TOLERANCE_PX;
		};
		wrapper.addEventListener('scroll', onScroll, { passive: true });
		return () => wrapper.removeEventListener('scroll', onScroll);
	});

	function handleClick() {
		if (!isClickable) return;
		const selection = window.getSelection();
		if (selection && selection.toString().length > 0) return;
		onToggleExpand?.();
	}
</script>

<div class="tool-chip-wrapper" class:has-code-block={showDetachedCode}>
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<span
		class="tool-chip"
		class:clickable={isClickable}
		class:expanded
		title={query || ''}
		onclick={handleClick}
		onkeydown={(e) => { if (isClickable && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onToggleExpand?.(); } }}
		role={isClickable ? 'button' : undefined}
		tabindex={isClickable ? 0 : undefined}
	>
		<span
			class="tool-chip-dot"
			class:executing={isExecuting}
			style:background-color={statusColor}
		></span>
		<IconComponent class="tool-chip-icon" size={14} />
		<span class="tool-chip-label">{displayLabel}</span>
		{#if query}
			<span class="tool-chip-query" class:query-expanded={expanded}>{query}</span>
		{/if}
		{#if inlineArgStream && argStream}
			<span class="tool-chip-query" class:query-expanded={expanded}>{argStream.text}</span>
		{/if}
	</span>

	{#if showDetachedCode && argStream}
		<div class="tool-chip-code-detached" bind:this={codeContainer}>
			<SyntaxHighlightedCode
				code={argStream.text || ' '}
				language="python"
				maxHeight="280px"
			/>
		</div>
	{/if}

	{#if imageArtifacts.length}
		<div class="tool-chip-artifacts">
			{#each imageArtifacts as artifact (artifact.name)}
				{@const src = artifact.url ?? `data:${artifact.mime};base64,${artifact.data_b64}`}
				<a
					href={src}
					target="_blank"
					rel="noopener noreferrer"
					class="artifact-link"
					aria-label="Open {artifact.name} in new tab"
				>
					<img
						{src}
						alt={artifact.name}
						class="artifact-thumb"
						loading="lazy"
					/>
				</a>
			{/each}
		</div>
	{/if}
</div>

<style>
	.tool-chip-wrapper {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.375rem;
		flex-basis: auto;
		min-width: 0;
		max-width: 100%;
	}

	/* When a code block lives below the chip, the wrapper must claim the
	   full row width — without this, the wrapper sizes to its widest
	   child (the long code lines), which means there's no constraint for
	   the inner overflow:auto to clip against, and the code visibly
	   escapes the bordered box. The `has-code-block` class scopes this
	   to chips that need it; web_search / weather / etc. without code
	   bodies stay at their natural pill width and can still wrap into
	   parallel-batch rows. */
	.tool-chip-wrapper.has-code-block {
		width: 100%;
	}

	.tool-chip {
		display: inline-flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.25rem 0.75rem;
		border-radius: 9999px;
		background: var(--muted);
		font-size: 0.8125rem;
		font-weight: 500;
		color: var(--muted-foreground);
		margin: 0.25rem 0.25rem 0.25rem 0;
		flex-basis: auto;
		flex-shrink: 0;
		max-width: 100%;
		transition: border-radius 150ms ease;
	}

	.tool-chip.clickable {
		cursor: pointer;
	}

	.tool-chip.clickable:hover {
		background: color-mix(in oklch, var(--muted) 80%, var(--foreground) 5%);
	}

	.tool-chip.expanded {
		border-radius: 0.75rem;
	}

	.tool-chip :global(.tool-chip-icon) {
		flex-shrink: 0;
		opacity: 0.7;
	}

	.tool-chip-label {
		line-height: 1;
		white-space: nowrap;
	}

	.tool-chip-query {
		line-height: 1;
		opacity: 0.7;
		font-weight: 400;
		font-style: italic;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
		max-width: 30ch;
	}

	.tool-chip-query.query-expanded {
		white-space: normal;
		max-width: none;
		word-break: break-word;
	}

	.tool-chip-code-detached {
		/* Sibling of .tool-chip — the chip stays a small pill while the
		   code lives in its own block below. align-self stretches us to
		   the row's content area (parent .tool-chip-wrapper is a flex
		   column with align-items: flex-start, so without this we'd be
		   the chip's natural width). min-width: 0 lets the inner code
		   shrink below its intrinsic content width when needed, which
		   is what prevents long Python lines from blowing out the
		   container before the wrapper's overflow:auto can kick in. */
		align-self: stretch;
		min-width: 0;
		width: 100%;
		max-width: 100%;
		margin: 0 0 0.25rem 0;
	}

	/* Tighten the SyntaxHighlightedCode wrapper inside our context:
	   smaller text, slimmer padding. The wrapper itself only needs
	   overflow-y for height clipping; horizontal scroll lives on the
	   <pre> below because that's where the inline <code> child renders
	   at its full content width. */
	.tool-chip-code-detached :global(.code-preview-wrapper) {
		font-size: 0.75rem;
		overflow-y: auto;
	}

	.tool-chip-code-detached :global(.code-preview-wrapper pre) {
		padding: 0.5rem 0.75rem;
		/* No overflow on the <pre>: its inner <code> is `display: block`
		   (set by .hljs) so it fits within the pre, and overflow at this
		   level would never observe the long lines escaping. */
	}

	/* The actual horizontal-scroll boundary lives on the <code> element.
	   highlight.js's theme stylesheet sets display:block but does not
	   include the `overflow-x: auto` rule from hljs's default.css, so
	   long lines (white-space: pre) extend past the code element's box
	   and visibly escape the bordered wrapper. Pinning overflow here is
	   what finally clips and scrolls. */
	.tool-chip-code-detached :global(.code-preview-wrapper code) {
		overflow-x: auto;
		max-width: 100%;
	}

	.tool-chip-dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.tool-chip-dot.executing {
		animation: pulse 1.5s ease-in-out infinite;
	}

	.tool-chip-artifacts {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		margin: 0 0 0.25rem 0.25rem;
	}

	.artifact-link {
		display: inline-block;
		line-height: 0;
		border-radius: 0.5rem;
		overflow: hidden;
		border: 1px solid var(--muted);
		transition: border-color 150ms ease;
	}

	.artifact-link:hover {
		border-color: color-mix(in oklch, var(--muted) 50%, var(--foreground) 15%);
	}

	.artifact-thumb {
		display: block;
		max-width: 280px;
		max-height: 280px;
		width: auto;
		height: auto;
		object-fit: contain;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
		}
		50% {
			opacity: 0.3;
		}
	}
</style>
