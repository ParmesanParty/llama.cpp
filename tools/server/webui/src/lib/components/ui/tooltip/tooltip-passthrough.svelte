<script lang="ts">
	import type { Snippet } from 'svelte';

	// No-op passthrough used for <Tooltip.Provider> and <Tooltip.Portal>.
	// This implementation has no shared provider context (each tooltip is
	// independent) and no need for explicit portal escape (popover top-layer
	// rendering escapes containing blocks natively). Call sites that wrap
	// children in these primitives continue to compile but the wrappers do
	// nothing beyond rendering their children inline. Any additional props
	// (delayDuration, skipDelayDuration, disableHoverableContent, etc.) are
	// silently ignored.
	let { children }: { children: Snippet; [key: string]: unknown } = $props();
</script>

{@render children()}
