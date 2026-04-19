<script lang="ts">
	import type { Snippet } from 'svelte';

	// Backward-compatible API. This implementation has no "trigger" element
	// of its own — the parent <Tooltip.Root> wrapper is the hover target,
	// and pointer listeners are attached by <Tooltip.Content> on mount. We
	// just render the user's content inline.
	//
	// Two call patterns are supported:
	//   1. <Tooltip.Trigger><button>...</button></Tooltip.Trigger>
	//      → renders `children` directly.
	//   2. <Tooltip.Trigger>{#snippet child({ props })}<button {...props}>...</button>{/snippet}</Tooltip.Trigger>
	//      → invokes the `child` snippet with an empty props object so
	//      existing bits-ui-style call sites still render their button. No
	//      event handlers need to be merged in because pointer events are
	//      handled at the <Tooltip.Root> level.
	let {
		children,
		child
	}: {
		children?: Snippet;
		child?: Snippet<[{ props: Record<string, unknown> }]>;
	} = $props();
</script>

{#if child}
	{@render child({ props: {} })}
{:else if children}
	{@render children()}
{/if}
