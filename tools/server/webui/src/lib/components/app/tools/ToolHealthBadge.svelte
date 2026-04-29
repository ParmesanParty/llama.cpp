<script lang="ts">
	import type { BreakerState } from '$lib/stores/toolHealth.svelte';

	interface Props {
		breaker_state?: BreakerState;
		preflight_ok?: boolean;
		recent_failures?: number;
	}

	let { breaker_state, preflight_ok = true, recent_failures = 0 }: Props = $props();

	const status = $derived.by<{ color: string; label: string; title: string } | null>(() => {
		if (breaker_state === undefined) return null;
		if (breaker_state === 'open') {
			return { color: 'bg-red-500', label: 'Open', title: 'Circuit breaker open — tool calls fail fast' };
		}
		if (breaker_state === 'half_open') {
			return { color: 'bg-amber-500', label: 'Recovering', title: 'Circuit breaker half-open — probing recovery' };
		}
		if (!preflight_ok || recent_failures > 0) {
			return { color: 'bg-amber-500', label: 'Degraded', title: 'Recent failures or preflight issues' };
		}
		return { color: 'bg-green-500', label: 'Healthy', title: 'Tool is healthy' };
	});
</script>

{#if status}
	<span class="inline-flex items-center gap-1.5" title={status.title}>
		<span class="h-2 w-2 rounded-full {status.color}" aria-hidden="true"></span>
		<span class="sr-only">{status.label}</span>
	</span>
{/if}
