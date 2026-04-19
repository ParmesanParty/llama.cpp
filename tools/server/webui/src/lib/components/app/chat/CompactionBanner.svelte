<script lang="ts">
	import { ChevronRight, History } from '@lucide/svelte';
	import { tick } from 'svelte';

	interface Props {
		summary: string;
		messageCount: number;
		summaryExpanded?: boolean;
		historyExpanded?: boolean;
		onToggleSummary?: () => void;
		onToggleHistory?: () => void;
	}

	let {
		summary,
		messageCount,
		summaryExpanded = false,
		historyExpanded = false,
		onToggleSummary,
		onToggleHistory
	}: Props = $props();

	let bannerEl: HTMLDivElement | undefined = $state();

	async function handleToggleHistory() {
		if (!bannerEl) {
			onToggleHistory?.();
			return;
		}

		// Capture banner position before DOM changes
		const rectBefore = bannerEl.getBoundingClientRect();
		const topBefore = rectBefore.top;

		onToggleHistory?.();

		// Wait for Svelte to update the DOM
		await tick();
		// Extra frame for the browser to lay out the new elements
		await new Promise((r) => requestAnimationFrame(r));

		// Restore banner to its previous visual position
		const rectAfter = bannerEl.getBoundingClientRect();
		const drift = rectAfter.top - topBefore;
		if (Math.abs(drift) > 2) {
			window.scrollBy({ top: drift, behavior: 'instant' });
		}
	}
</script>

<div class="mx-auto w-full max-w-[48rem] my-4" bind:this={bannerEl}>
	<button
		class="w-full flex items-center gap-2 px-4 py-3 rounded-lg
			   bg-base-200/50 hover:bg-base-200/80 transition-colors
			   text-sm text-base-content/60 cursor-pointer"
		onclick={() => onToggleSummary?.()}
	>
		<span
			class="transition-transform duration-200"
			class:rotate-90={summaryExpanded}
		>
			<ChevronRight size={16} />
		</span>
		<span class="font-medium">Conversation summary</span>
		<span class="text-xs opacity-60">({messageCount} messages compacted)</span>
	</button>

	{#if summaryExpanded}
		<div class="mt-2 px-4 py-3 rounded-lg bg-base-200/30 text-sm
					prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap">
			{summary}
		</div>

		<button
			class="mt-2 flex items-center gap-1.5 px-4 py-2 text-xs
				   text-base-content/40 hover:text-base-content/60 transition-colors cursor-pointer"
			onclick={handleToggleHistory}
		>
			<History size={12} />
			{#if historyExpanded}
				Hide original messages
			{:else}
				Show original messages
			{/if}
		</button>
	{/if}
</div>
