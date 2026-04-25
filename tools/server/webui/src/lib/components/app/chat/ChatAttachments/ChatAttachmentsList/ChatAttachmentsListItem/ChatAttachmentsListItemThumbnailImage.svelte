<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import { ImageIcon, ImageOffIcon, X } from '@lucide/svelte';

	interface Props {
		class?: string;
		height?: string;
		id: string;
		imageClass?: string;
		onclick?: (event?: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		preview: string;
		readonly?: boolean;
		width?: string;
		// Optional pixel dimensions for aspect-ratio reservation. When present,
		// the wrapper uses `aspect-ratio: pixelWidth / pixelHeight` so the row
		// reserves correctly-proportioned space before pixels load — avoiding
		// CLS as base64 thumbnails decode.
		pixelWidth?: number;
		pixelHeight?: number;
	}

	let {
		class: className = '',
		height = 'h-16',
		id,
		imageClass = '',
		onclick,
		onRemove,
		name,
		preview,
		readonly = false,
		width = 'w-auto',
		pixelWidth,
		pixelHeight
	}: Props = $props();

	type Status = 'loading' | 'loaded' | 'error';
	let status = $state<Status>('loading');

	let aspectRatioStyle = $derived.by(() => {
		if (pixelWidth && pixelHeight) {
			return `aspect-ratio: ${pixelWidth} / ${pixelHeight};`;
		}
		return undefined;
	});

	function handleLoad() {
		status = 'loaded';
	}
	function handleError() {
		status = 'error';
	}
</script>

{#snippet image()}
	<img
		src={preview}
		alt={name}
		class="{height} {width} cursor-pointer object-cover {imageClass}"
		width={pixelWidth}
		height={pixelHeight}
		loading="lazy"
		onload={handleLoad}
		onerror={handleError}
	/>
{/snippet}

<div
	class="group relative overflow-hidden rounded-lg bg-muted shadow-lg dark:border dark:border-muted {className}"
	style={aspectRatioStyle}
>
	{#if onclick}
		<button
			aria-label="Preview {name}"
			class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			{onclick}
			type="button"
		>
			{@render image()}
		</button>
	{:else}
		{@render image()}
	{/if}

	{#if status === 'loading'}
		<div
			class="pointer-events-none absolute inset-0 flex items-center justify-center bg-muted/40"
		>
			<ImageIcon class="h-6 w-6 animate-pulse text-muted-foreground/50" />
		</div>
	{:else if status === 'error'}
		<div
			class="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-muted p-2 text-center text-xs text-muted-foreground"
		>
			<ImageOffIcon class="h-5 w-5" />
			<span>Image unavailable</span>
		</div>
	{/if}

	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<ActionIcon
				class="text-white"
				icon={X}
				onclick={() => onRemove?.(id)}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>
