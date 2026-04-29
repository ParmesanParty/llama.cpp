<script lang="ts">
	import { cn } from '$lib/components/ui/utils.js';
	import type { Snippet } from 'svelte';

	// Backward-compatible API. We accept the bits-ui ContentProps surface but
	// only use `side`, `align`, and `class`. The `sideOffset` prop is accepted
	// (so call sites compile) but ignored — we use a fixed 6px gap between
	// trigger and tooltip.
	let {
		class: className,
		side = 'top',
		align = 'center',
		children
	}: {
		class?: string;
		side?: 'top' | 'bottom' | 'left' | 'right';
		align?: 'start' | 'center' | 'end';
		sideOffset?: number;
		children?: Snippet;
		[key: string]: unknown;
	} = $props();

	// Implementation strategy: render this tooltip into the browser's TOP
	// LAYER via the native HTML `popover` attribute. The top layer is a
	// special render layer above the page that ignores `overflow: hidden`,
	// `transform`, `clip-path`, and every other clipping mechanism — so the
	// tooltip can never be clipped by an ancestor (e.g. a DropdownMenu's
	// `overflow-x-hidden`). This is the same outcome bits-ui's Portal
	// achieved, but native and JS-free at the framework level.
	//
	// Hover state is driven by a tiny per-tooltip pointer listener attached
	// to the trigger element (the nearest ancestor with the
	// `llm-tooltip-trigger` class — applied by <Tooltip.Root>). On
	// pointerenter we measure the trigger's bounding rect, position the
	// popover via inline `left`/`top` (the popover uses `position: fixed`),
	// and call `showPopover()`. On pointerleave we call `hidePopover()`.
	// Each tooltip is independent — no shared coordinator, so the bits-ui
	// flicker bug cannot occur.
	const GAP = 6;
	let popoverEl: HTMLElement | undefined = $state();
	let trigger: HTMLElement | null = null;

	function position() {
		if (!popoverEl || !trigger) return;
		const r = trigger.getBoundingClientRect();
		const pw = popoverEl.offsetWidth;
		const ph = popoverEl.offsetHeight;
		let left = 0;
		let top = 0;
		if (side === 'top') {
			top = r.top - ph - GAP;
			left =
				align === 'start'
					? r.left
					: align === 'end'
						? r.right - pw
						: r.left + r.width / 2 - pw / 2;
		} else if (side === 'bottom') {
			top = r.bottom + GAP;
			left =
				align === 'start'
					? r.left
					: align === 'end'
						? r.right - pw
						: r.left + r.width / 2 - pw / 2;
		} else if (side === 'left') {
			left = r.left - pw - GAP;
			top =
				align === 'start'
					? r.top
					: align === 'end'
						? r.bottom - ph
						: r.top + r.height / 2 - ph / 2;
		} else {
			// right
			left = r.right + GAP;
			top =
				align === 'start'
					? r.top
					: align === 'end'
						? r.bottom - ph
						: r.top + r.height / 2 - ph / 2;
		}
		// Clamp to viewport so a tooltip can never extend off-screen even if
		// the chosen side has insufficient room.
		const margin = 4;
		left = Math.max(margin, Math.min(left, window.innerWidth - pw - margin));
		top = Math.max(margin, Math.min(top, window.innerHeight - ph - margin));
		popoverEl.style.left = `${left}px`;
		popoverEl.style.top = `${top}px`;
	}

	function show() {
		if (!popoverEl) return;
		try {
			popoverEl.showPopover();
		} catch {
			// Already shown or unsupported — ignore.
		}
		// Position after show so offsetWidth/offsetHeight are populated.
		position();
	}

	function hide() {
		if (!popoverEl) return;
		try {
			popoverEl.hidePopover();
		} catch {
			// Already hidden — ignore.
		}
	}

	function attach(el: HTMLElement) {
		// Walk up to the nearest <Tooltip.Root> wrapper (marked with the
		// `llm-tooltip-trigger` class) and attach pointer listeners to it.
		// `closest` walks up including el's parent, so this works regardless
		// of intermediate wrappers inserted by call sites.
		trigger = el.parentElement?.closest('.llm-tooltip-trigger') as HTMLElement | null;
		if (!trigger) return;
		trigger.addEventListener('pointerenter', show);
		trigger.addEventListener('pointerleave', hide);
		// Reposition on scroll/resize while visible (handles pages that
		// scroll under the tooltip after it's shown).
		window.addEventListener('scroll', position, true);
		window.addEventListener('resize', position);
		return () => {
			trigger?.removeEventListener('pointerenter', show);
			trigger?.removeEventListener('pointerleave', hide);
			window.removeEventListener('scroll', position, true);
			window.removeEventListener('resize', position);
		};
	}

	$effect(() => {
		if (popoverEl) {
			return attach(popoverEl);
		}
	});
</script>

<!--
	`popover="manual"` puts the element in the browser top layer when shown
	via showPopover(). Top-layer rendering escapes every kind of clipping
	(overflow, transform, clip-path), so the tooltip cannot be cut off by an
	ancestor like a DropdownMenu with overflow-x-hidden.

	`position: fixed` is set inline so it survives the top-layer rendering;
	`left`/`top` are written by the position() function on show.
-->
<span
	bind:this={popoverEl}
	popover="manual"
	role="tooltip"
	style="position: fixed; margin: 0;"
	class={cn(
		'z-50 w-fit whitespace-nowrap rounded-md border-0 bg-primary px-3 py-1.5 text-xs text-balance text-primary-foreground',
		className
	)}
>
	{@render children?.()}
</span>
