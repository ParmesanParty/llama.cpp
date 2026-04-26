<script lang="ts">
	import { RefreshCw, Activity } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { toolHealthStore } from '$lib/stores/toolHealth.svelte';
	import ToolCard from './ToolCard.svelte';

	const sortedTools = $derived(
		Array.from(toolHealthStore.tools.values()).sort((a, b) =>
			(a.display_name || a.name).localeCompare(b.display_name || b.name)
		)
	);

	function formatUptime(sec: number): string {
		if (sec < 60) return `${Math.floor(sec)}s`;
		if (sec < 3600) return `${Math.floor(sec / 60)}m`;
		const h = Math.floor(sec / 3600);
		const m = Math.floor((sec % 3600) / 60);
		return `${h}h ${m}m`;
	}

	async function handleRefresh() {
		await toolHealthStore.fetchSnapshot();
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<div>
			<h3 class="text-base font-semibold">Server-side tools</h3>
			<p class="text-xs text-muted-foreground">
				Tools the model can invoke during a chat. Status updates live.
			</p>
		</div>
		<Button variant="outline" size="sm" onclick={handleRefresh} disabled={toolHealthStore.loading}>
			<RefreshCw class="mr-1.5 h-3 w-3 {toolHealthStore.loading ? 'animate-spin' : ''}" />
			Refresh
		</Button>
	</div>

	{#if toolHealthStore.server}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Activity class="h-3 w-3" />
			<span>Server: {toolHealthStore.server.state}</span>
			<span>·</span>
			<span>uptime {formatUptime(toolHealthStore.server.uptime_seconds)}</span>
			<span>·</span>
			<span>{toolHealthStore.server.active_requests} active</span>
		</div>
	{/if}

	{#if toolHealthStore.error}
		<div class="rounded-md border border-dashed border-destructive/50 bg-destructive/5 p-4 text-sm">
			<p class="text-destructive">Failed to load tool health: {toolHealthStore.error}</p>
			<Button variant="outline" size="sm" class="mt-2" onclick={handleRefresh}>Retry</Button>
		</div>
	{:else if toolHealthStore.tools.size === 0 && !toolHealthStore.loading}
		<div class="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
			No server-side tools registered.
		</div>
	{:else if toolHealthStore.loading && toolHealthStore.tools.size === 0}
		<div class="space-y-3">
			{#each Array(3) as _}
				<div class="h-24 animate-pulse rounded-md border bg-muted/30"></div>
			{/each}
		</div>
	{:else}
		<div class="space-y-3">
			{#each sortedTools as tool (tool.name)}
				<ToolCard entry={tool} />
			{/each}
		</div>
	{/if}
</div>
