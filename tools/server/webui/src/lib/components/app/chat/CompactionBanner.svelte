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

	// Toggling history changes the heights of pre-banner messages, which
	// would scroll the banner out of view.  Capture the banner's viewport
	// position before the DOM mutates, wait for layout, then nudge the
	// scroll position so the banner stays visually still.
	async function handleToggleHistory() {
		if (!bannerEl) {
			onToggleHistory?.();
			return;
		}

		const rectBefore = bannerEl.getBoundingClientRect();
		const topBefore = rectBefore.top;

		onToggleHistory?.();

		await tick();
		await new Promise((r) => requestAnimationFrame(r));

		const rectAfter = bannerEl.getBoundingClientRect();
		const drift = rectAfter.top - topBefore;
		if (Math.abs(drift) > 2) {
			window.scrollBy({ top: drift, behavior: 'instant' });
		}
	}
</script>

<div class="mx-auto my-4 w-full max-w-[48rem]" bind:this={bannerEl}>
	<button
		class="flex w-full cursor-pointer items-center gap-2 rounded-lg bg-muted/40 px-4 py-3 text-sm text-muted-foreground transition-colors hover:bg-muted/60"
		onclick={() => onToggleSummary?.()}
	>
		<span class="transition-transform duration-200" class:rotate-90={summaryExpanded}>
			<ChevronRight size={16} />
		</span>
		<span class="font-medium">Conversation summary</span>
		<span class="text-xs opacity-60">({messageCount} messages compacted)</span>
	</button>

	{#if summaryExpanded}
		<div
			class="prose prose-sm dark:prose-invert mt-2 max-w-none rounded-lg bg-muted/30 px-4 py-3 text-sm whitespace-pre-wrap"
		>
			{summary}
		</div>

		<button
			class="mt-2 flex cursor-pointer items-center gap-1.5 px-4 py-2 text-xs text-muted-foreground/60 transition-colors hover:text-muted-foreground"
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
