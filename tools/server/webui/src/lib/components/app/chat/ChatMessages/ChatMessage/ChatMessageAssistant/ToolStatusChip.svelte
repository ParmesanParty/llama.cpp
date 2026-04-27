<script lang="ts">
	import { Search, Cloud, Code, Image, Globe } from '@lucide/svelte';
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

	let IconComponent = $derived(iconMap[tool] ?? Globe);
	let label = $derived(labelMap[tool] ?? tool);
	let isWriting = $derived(
		!!argStream && !argStream.complete && status !== 'completed' && status !== 'failed'
	);
	let writingLabel = $derived(writingLabelMap[tool] ?? 'Writing');
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

	function handleClick() {
		if (!isClickable) return;
		const selection = window.getSelection();
		if (selection && selection.toString().length > 0) return;
		onToggleExpand?.();
	}
</script>

<div class="tool-chip-wrapper">
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
		{#if argStream && !query}
			{#if tool === 'code_exec'}
				<span
					class="tool-chip-arg-code"
					class:expanded
				>{argStream.text || ' '}</span>
			{:else}
				<span class="tool-chip-query" class:query-expanded={expanded}>{argStream.text}</span>
			{/if}
		{/if}
	</span>

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

	.tool-chip-arg-code {
		display: block;
		font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
		font-size: 0.75rem;
		line-height: 1.4;
		max-height: 0;
		max-width: 0;
		overflow: hidden;
		white-space: pre;
		opacity: 0.85;
		transition: max-height 200ms ease, max-width 200ms ease, padding 200ms ease;
		padding: 0 0;
	}

	.tool-chip-arg-code.expanded {
		max-height: 24rem;
		max-width: 100%;
		overflow: auto;
		padding: 0.5rem 0.75rem 0.25rem;
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
