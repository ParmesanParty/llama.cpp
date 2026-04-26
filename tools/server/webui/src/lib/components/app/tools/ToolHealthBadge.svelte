<script lang="ts">
	import type { BadgeColor } from '$lib/stores/toolHealth.svelte';

	interface Props {
		color: BadgeColor;
		label?: string;
		title?: string;
	}

	let { color, label, title }: Props = $props();

	const colorClass = $derived.by(() => {
		if (color === 'green') return 'bg-green-500';
		if (color === 'amber') return 'bg-amber-500';
		if (color === 'gray') return 'bg-muted-foreground/40';
		return 'bg-red-500';
	});
</script>

<span class="inline-flex items-center gap-1.5" {title}>
	<span class="h-2 w-2 rounded-full {colorClass}" aria-hidden="true"></span>
	{#if label}
		<span class="text-xs">{label}</span>
	{/if}
</span>
