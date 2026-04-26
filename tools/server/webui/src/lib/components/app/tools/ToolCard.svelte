<script lang="ts">
	import { ChevronDown, ChevronRight, RotateCcw, Search, Image, Cloud, Terminal, Link, Wrench } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import ToolHealthBadge from './ToolHealthBadge.svelte';
	import { toolHealthStore, type ToolHealthEntry } from '$lib/stores/toolHealth.svelte';

	interface Props {
		entry: ToolHealthEntry;
	}

	let { entry }: Props = $props();
	let paramsExpanded = $state(false);
	let resetting = $state(false);

	const ICON_BY_TOOL: Record<string, typeof Wrench> = {
		web_search: Search,
		url_fetch: Link,
		weather: Cloud,
		code_exec: Terminal,
		image_gen: Image,
	};
	const Icon = $derived(ICON_BY_TOOL[entry.name] ?? Wrench);

	const status = $derived.by<{ color: 'green' | 'amber' | 'red'; label: string }>(() => {
		if (entry.breaker_state === 'open') return { color: 'red', label: 'Open' };
		if (entry.breaker_state === 'half_open') return { color: 'amber', label: 'Recovering' };
		if (!entry.preflight_ok || entry.recent_failures > 0) return { color: 'amber', label: 'Degraded' };
		return { color: 'green', label: 'Healthy' };
	});

	const lastEventLine = $derived.by<string | null>(() => {
		if (!entry.preflight_ok && entry.preflight_reason) {
			return `Preflight failed: ${entry.preflight_reason}`;
		}
		const parts: string[] = [];
		if (entry.last_success) parts.push(`Last call: ${formatRel(entry.last_success)} ago`);
		if (entry.recent_failures > 0) parts.push(`${entry.recent_failures} recent failure(s)`);
		else if (entry.last_failure && !entry.last_success) parts.push(`Last failure: ${formatRel(entry.last_failure)} ago`);
		return parts.length ? parts.join(' · ') : null;
	});

	function formatRel(unix: number): string {
		const sec = Math.max(0, Math.floor(Date.now() / 1000 - unix));
		if (sec < 60) return `${sec}s`;
		if (sec < 3600) return `${Math.floor(sec / 60)}m`;
		if (sec < 86400) return `${Math.floor(sec / 3600)}h`;
		return `${Math.floor(sec / 86400)}d`;
	}

	const canReset = $derived(entry.breaker_state !== 'closed');

	async function handleReset() {
		resetting = true;
		try {
			await toolHealthStore.resetBreaker(entry.name);
		} finally {
			resetting = false;
		}
	}

	const paramsList = $derived.by<Array<{ name: string; type: string; required: boolean; description: string }>>(() => {
		const props = (entry.parameters?.properties ?? {}) as Record<string, { type?: string; description?: string }>;
		const required = new Set(((entry.parameters?.required ?? []) as string[]));
		return Object.entries(props).map(([name, schema]) => ({
			name,
			type: schema.type ?? 'any',
			required: required.has(name),
			description: schema.description ?? '',
		}));
	});
</script>

<Card.Root class="space-y-2 p-4">
	<div class="flex items-start justify-between gap-3">
		<div class="flex min-w-0 items-start gap-2">
			<Icon class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
			<div class="min-w-0">
				<div class="flex flex-wrap items-center gap-2">
					<span class="truncate font-medium">{entry.display_name || entry.name}</span>
					<ToolHealthBadge color={status.color} label={status.label} />
				</div>
				{#if entry.summary}
					<p class="mt-1 text-sm text-muted-foreground">{entry.summary}</p>
				{/if}
			</div>
		</div>
	</div>

	{#if lastEventLine}
		<p class="text-xs text-muted-foreground">{lastEventLine}</p>
	{/if}

	<div class="flex flex-wrap items-center gap-2 pt-1">
		{#if canReset}
			<Button
				variant="outline"
				size="sm"
				disabled={resetting}
				onclick={handleReset}
			>
				<RotateCcw class="mr-1.5 h-3 w-3" />
				{resetting ? 'Resetting…' : 'Reset breaker'}
			</Button>
		{/if}
		{#if paramsList.length > 0}
			<Button
				variant="ghost"
				size="sm"
				onclick={() => (paramsExpanded = !paramsExpanded)}
			>
				{#if paramsExpanded}
					<ChevronDown class="mr-1.5 h-3 w-3" />
				{:else}
					<ChevronRight class="mr-1.5 h-3 w-3" />
				{/if}
				{paramsExpanded ? 'Hide' : 'Show'} parameters
			</Button>
		{/if}
	</div>

	{#if paramsExpanded && paramsList.length > 0}
		<div class="mt-2 border-t border-border/30 pt-2">
			<table class="w-full text-xs">
				<tbody>
					{#each paramsList as p (p.name)}
						<tr class="align-top">
							<td class="whitespace-nowrap py-1 pr-2 font-mono">
								{p.name}{#if p.required}<span class="text-amber-500">*</span>{/if}
							</td>
							<td class="whitespace-nowrap py-1 pr-2 text-muted-foreground">{p.type}</td>
							<td class="py-1 text-muted-foreground">{p.description}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</Card.Root>
