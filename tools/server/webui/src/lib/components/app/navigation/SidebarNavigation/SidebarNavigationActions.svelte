<script lang="ts">
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import type { Component } from 'svelte';
	import { SearchInput } from '$lib/components/app';
	import { page } from '$app/state';
	import { SIDEBAR_ACTIONS_ITEMS } from '$lib/constants/ui';
	import { toolHealthStore } from '$lib/stores/toolHealth.svelte';

	const toolHealthDot = $derived.by<string | null>(() => {
		const c = toolHealthStore.badgeColor;
		if (c === 'green') return null;
		if (c === 'amber') return 'bg-amber-500';
		if (c === 'red') return 'bg-red-500';
		return 'bg-muted-foreground/40';
	});

	interface Props {
		handleMobileSidebarItemClick: () => void;
		isSearchModeActive: boolean;
		searchQuery: string;
		isCancelAlwaysVisible?: boolean;
		onSearchDeactivated?: () => void;
	}

	let {
		handleMobileSidebarItemClick,
		isSearchModeActive = $bindable(),
		searchQuery = $bindable(),
		isCancelAlwaysVisible = false,
		onSearchDeactivated
	}: Props = $props();

	let searchInputRef = $state<HTMLInputElement | null>(null);

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
		onSearchDeactivated?.();
	}

	export function activateSearch() {
		isSearchModeActive = true;
		// Focus after Svelte renders the input
		queueMicrotask(() => searchInputRef?.focus());
	}
</script>

{#snippet itemIcon(IconComponent: Component)}
	<IconComponent class="h-4 w-4" />
{/snippet}

<div class="my-1 space-y-1">
	{#if isSearchModeActive}
		<SearchInput
			bind:value={searchQuery}
			bind:ref={searchInputRef}
			onClose={handleSearchModeDeactivate}
			onKeyDown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
			placeholder="Search conversations..."
			{isCancelAlwaysVisible}
		/>
	{:else}
		{#each SIDEBAR_ACTIONS_ITEMS as item (item.route)}
			{#if !item.route}
				<Button
					class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100"
					onclick={activateSearch}
					variant="ghost"
				>
					<div class="flex items-center gap-2">
						{@render itemIcon(item.icon)}

						{item.tooltip}
					</div>

					{#if item.keys}
						<KeyboardShortcutInfo keys={item.keys} />
					{/if}
				</Button>
			{:else}
				<Button
					class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {(item.activeRouteId &&
						page.route.id === item.activeRouteId) ||
					(item.activeRoutePrefix && page.route.id?.startsWith(item.activeRoutePrefix))
						? 'bg-accent text-accent-foreground'
						: ''}"
					href={item.route}
					onclick={handleMobileSidebarItemClick}
					variant="ghost"
				>
					<div class="flex items-center gap-2">
						{@render itemIcon(item.icon)}

						{item.tooltip}

						{#if item.id === 'tools' && toolHealthDot}
							<span
								class="ml-1 inline-block h-2 w-2 rounded-full {toolHealthDot}"
								aria-hidden="true"
							></span>
						{/if}
					</div>

					{#if item.keys}
						<KeyboardShortcutInfo keys={item.keys} />
					{/if}
				</Button>
			{/if}
		{/each}
	{/if}
</div>
