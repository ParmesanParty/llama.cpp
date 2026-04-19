<script lang="ts">
	import { Search, Cloud, Code, Image, Globe } from '@lucide/svelte';

	interface Props {
		tool: string;
		status: string;
		query?: string;
		expanded?: boolean;
		onToggleExpand?: () => void;
	}

	let { tool, status, query, expanded = false, onToggleExpand }: Props = $props();

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

	let IconComponent = $derived(iconMap[tool] ?? Globe);
	let label = $derived(labelMap[tool] ?? tool);
	let statusColor = $derived(
		status === 'completed'
			? 'var(--green-500, #22c55e)'
			: status === 'failed'
				? 'var(--red-500, #ef4444)'
				: 'var(--amber-500, #f59e0b)'
	);
	let isExecuting = $derived(status === 'executing');
	let isClickable = $derived(!!query && !!onToggleExpand);

	function handleClick() {
		if (!isClickable) return;
		const selection = window.getSelection();
		if (selection && selection.toString().length > 0) return;
		onToggleExpand?.();
	}
</script>

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
	<span class="tool-chip-label">{label}</span>
	{#if query}
		<span class="tool-chip-query" class:query-expanded={expanded}>{query}</span>
	{/if}
</span>

<style>
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

	.tool-chip-dot {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 50%;
		flex-shrink: 0;
	}

	.tool-chip-dot.executing {
		animation: pulse 1.5s ease-in-out infinite;
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
