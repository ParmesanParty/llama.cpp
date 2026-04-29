<script lang="ts">
	import { ChevronDown, Loader2, Package } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { modelsStore, switching, activePreset } from '$lib/stores/models.svelte';
	import {
		MODEL_QUANTIZATION_SEGMENT_RE,
		MODEL_CUSTOM_QUANTIZATION_PREFIX_RE
	} from '$lib/constants';
	import { DropdownMenuSearchable } from '$lib/components/app';
	import type { PresetInfo } from '$lib/types/api';

	// Drop the trailing quant token ("Q4_K_L", "UD-IQ4_NL_XL", etc.) from a preset
	// friendly name so the collapsed trigger stays readable; full name remains in
	// the dropdown list and in the tooltip.
	function stripTrailingQuant(name: string): string {
		const trimmed = name.trim();
		if (!trimmed) return name;

		const lastSpace = trimmed.lastIndexOf(' ');
		if (lastSpace === -1) return name;

		const head = trimmed.slice(0, lastSpace);
		const tail = trimmed.slice(lastSpace + 1);

		if (MODEL_QUANTIZATION_SEGMENT_RE.test(tail)) return head;

		const dashIdx = tail.indexOf('-');
		if (dashIdx > 0) {
			const prefix = tail.slice(0, dashIdx);
			const rest = tail.slice(dashIdx + 1);
			if (
				MODEL_CUSTOM_QUANTIZATION_PREFIX_RE.test(prefix) &&
				MODEL_QUANTIZATION_SEGMENT_RE.test(rest)
			) {
				return head;
			}
		}

		return name;
	}

	interface Props {
		class?: string;
		disabled?: boolean;
		forceForegroundText?: boolean;
	}

	let {
		class: className = '',
		disabled = false,
		forceForegroundText = false
	}: Props = $props();

	let isSwitching = $derived(switching());
	let currentPreset = $derived(activePreset());
	let presets = $derived(modelsStore.presets);

	let presetEntries = $derived(
		Object.entries(presets).sort(([a], [b]) => a.localeCompare(b))
	);

	let searchTerm = $state('');
	let isOpen = $state(false);

	let filteredEntries = $derived(
		searchTerm
			? presetEntries.filter(([name, info]: [string, PresetInfo]) =>
					name.toLowerCase().includes(searchTerm.toLowerCase()) ||
					info.friendly_name.toLowerCase().includes(searchTerm.toLowerCase())
				)
			: presetEntries
	);

	let sortedEntries = $derived.by(() => {
		const cur = currentPreset;
		if (!cur) return filteredEntries;
		const active: typeof filteredEntries = [];
		const rest: typeof filteredEntries = [];
		for (const entry of filteredEntries) {
			(entry[0] === cur ? active : rest).push(entry);
		}
		return [...active, ...rest];
	});

	let activeInfo = $derived.by(() => {
		if (!currentPreset) return null;
		return presets[currentPreset] ?? null;
	});

	function handleOpenChange(open: boolean) {
		isOpen = open;
		if (open) {
			searchTerm = '';
		}
	}

	export function open() {
		handleOpenChange(true);
	}

	async function handleSelect(presetName: string) {
		if (presetName === currentPreset) {
			handleOpenChange(false);
			return;
		}
		const info = presets[presetName];
		if (!info?.downloaded) return;

		handleOpenChange(false);
		try {
			await modelsStore.switchModel(presetName);
		} catch {
			// Error handled by store (toast)
		}
	}
</script>

<div class={cn('relative inline-flex flex-col items-end gap-1', className)}>
	<DropdownMenu.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<DropdownMenu.Trigger disabled={disabled || isSwitching}>
			<button
				type="button"
				class={cn(
					'inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60',
					forceForegroundText ? 'text-foreground' : 'text-muted-foreground',
					isOpen ? 'text-foreground' : ''
				)}
				style="max-width: min(calc(100cqw - 9rem), 20rem)"
				disabled={disabled || isSwitching}
			>
				<Package class="h-3.5 w-3.5 shrink-0" />

				{#if activeInfo}
					<Tooltip.Root>
						<Tooltip.Trigger class="min-w-0 overflow-hidden">
							<span class="block truncate text-xs font-medium">
								{stripTrailingQuant(activeInfo.friendly_name)}
							</span>
						</Tooltip.Trigger>
						<Tooltip.Content>
							<p class="font-mono text-xs">{currentPreset}</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{:else}
					<span class="min-w-0 truncate font-medium">Select model</span>
				{/if}

				{#if isSwitching}
					<Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
				{:else}
					<ChevronDown class="h-3 w-3.5 shrink-0" />
				{/if}
			</button>
		</DropdownMenu.Trigger>

		<DropdownMenu.Content
			align="end"
			class="w-full max-w-[100vw] pt-0 sm:w-max sm:min-w-[20rem] sm:max-w-[calc(100vw-2rem)]"
		>
			<DropdownMenuSearchable
				bind:searchValue={searchTerm}
				placeholder="Search models..."
				emptyMessage="No models found."
				isEmpty={filteredEntries.length === 0}
			>
				<div class="max-h-[min(24rem,50vh)] overflow-y-auto">
					{#each sortedEntries as [name, info] (name)}
						{@const isActive = name === currentPreset}
						<button
							type="button"
							class={cn(
								'group flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition',
								info.downloaded
									? 'cursor-pointer hover:bg-accent hover:text-accent-foreground'
									: 'cursor-not-allowed opacity-50',
								isActive ? 'bg-accent/50' : ''
							)}
							disabled={!info.downloaded || isSwitching}
							onclick={() => handleSelect(name)}
						>
							<div class="min-w-0 flex-1">
								<div class="truncate font-medium">{info.friendly_name}</div>
								<div class="text-xs text-muted-foreground">
									{info.size_approx}
									{#if info.ctx_size}
										· {Math.round(info.ctx_size / 1024)}K ctx
									{/if}
									{#if !info.downloaded}
										· Not downloaded
									{/if}
								</div>
							</div>

							<div class="flex w-4 shrink-0 items-center justify-center">
								{#if isActive}
									<span class="h-2 w-2 rounded-full bg-green-500"></span>
								{:else if !info.downloaded}
									<span class="h-2 w-2 rounded-full bg-muted-foreground/30"></span>
								{/if}
							</div>
						</button>
					{/each}
				</div>
			</DropdownMenuSearchable>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
</div>
