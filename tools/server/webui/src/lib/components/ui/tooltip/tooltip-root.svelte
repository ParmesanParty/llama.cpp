<script lang="ts">
	import type { Snippet } from 'svelte';

	// Backward-compatible API: accept any prop bits-ui's Tooltip.Root accepted
	// (delayDuration, disableHoverableContent, open, etc.) so call sites don't
	// need to change. We ignore them — there are no shared timers or context
	// in this implementation.
	let { children }: { children: Snippet; [key: string]: unknown } = $props();
</script>

<!--
	This wrapper span serves two purposes:

	1. It is the hover target for the tooltip. `inline-flex` sizes it to its
	   child content (typically a button), so pointerenter/leave events fire
	   only on the trigger surface — not on the gap between buttons.

	2. The `llm-tooltip-trigger` class is a marker that <Tooltip.Content>
	   uses to find its trigger element. The popover-based content walks up
	   via `closest('.llm-tooltip-trigger')` to attach pointer listeners and
	   compute its position from the trigger's bounding rect on hover. This
	   class has no styles attached — it exists purely as a query target.
-->
<span class="llm-tooltip-trigger inline-flex">
	{@render children()}
</span>
