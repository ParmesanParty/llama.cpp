<script lang="ts">
	import type { SourceItem } from '$lib/types/chat';

	interface Props {
		sources: SourceItem[];
	}

	let { sources }: Props = $props();

	const PREVIEW_COUNT = 3;
	let expanded = $state(false);

	// Display number matches citation numbering:
	// 0-indexed sources (index 0,1,2) → display [1],[2],[3] (model writes [1] for index 0)
	// The proxy always sends 0-indexed; the display offset is always +1.
	const displayIndex = (source: SourceItem) => source.index + 1;

	let previewSources = $derived(sources.slice(0, PREVIEW_COUNT));
	let hasMore = $derived(sources.length > PREVIEW_COUNT);
	let visibleSources = $derived(expanded ? sources : previewSources);
</script>

{#if sources.length > 0}
	<div class="sources-footer">
		<button class="sources-header" onclick={() => expanded = !expanded}>
			<span class="sources-label">{sources.length} Source{sources.length !== 1 ? 's' : ''}</span>
			<span class="sources-toggle">
				{#if hasMore}
					{expanded ? 'Collapse' : `Show all ${sources.length}`}
				{:else}
					{expanded ? 'Collapse' : 'Expand'}
				{/if}
				<svg class="chevron" class:expanded width="12" height="12" viewBox="0 0 12 12" fill="none">
					<path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
				</svg>
			</span>
		</button>

		<ol class="sources-list">
			{#each visibleSources as source (source.index)}
				<li value={displayIndex(source)}>
					<a href={source.url} target="_blank" rel="noopener noreferrer">
						{source.title || source.url}
					</a>
				</li>
			{/each}
		</ol>

		{#if hasMore}
			{#if expanded}
				<button class="show-more" onclick={() => expanded = false}>
					Show fewer
				</button>
			{:else}
				<button class="show-more" onclick={() => expanded = true}>
					+{sources.length - PREVIEW_COUNT} more source{sources.length - PREVIEW_COUNT !== 1 ? 's' : ''}
				</button>
			{/if}
		{/if}
	</div>
{/if}

<style>
	.sources-footer {
		margin-top: 1.5rem;
		padding-top: 1rem;
		border-top: 1px solid var(--border);
	}

	.sources-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		background: none;
		border: none;
		padding: 0;
		margin-bottom: 0.5rem;
		cursor: pointer;
		color: inherit;
		font: inherit;
	}

	.sources-label {
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--muted-foreground);
	}

	.sources-toggle {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		font-size: 0.75rem;
		color: var(--muted-foreground);
	}

	.sources-header:hover .sources-toggle {
		color: var(--primary);
	}

	.chevron {
		transition: transform 0.2s ease;
	}

	.chevron.expanded {
		transform: rotate(180deg);
	}

	.sources-list {
		list-style-type: decimal;
		margin-left: 1.25rem;
		font-size: 0.875rem;
		line-height: 1.75;
	}

	.sources-list li {
		color: var(--muted-foreground);
	}

	.sources-list a {
		color: var(--primary);
		text-decoration: none;
		text-underline-offset: 2px;
	}

	.sources-list a:hover {
		text-decoration: underline;
	}

	.show-more {
		background: none;
		border: none;
		padding: 0.25rem 0;
		margin-left: 1.25rem;
		font-size: 0.8rem;
		color: var(--muted-foreground);
		cursor: pointer;
	}

	.show-more:hover {
		color: var(--primary);
	}
</style>
