<script lang="ts">
	import SettingsChatFooter from './SettingsChatFooter.svelte';
	import SettingsChatFields from './SettingsChatFields.svelte';
	import SettingsChatToolsTab from './SettingsChatToolsTab.svelte';
	import SettingsChatImportExportTab from './SettingsChatImportExportTab.svelte';
	import {
		SettingsChatDesktopSidebar,
		SettingsChatMobileHeader
	} from '$lib/components/app/settings';
	import { Brain, ChevronDown, RotateCcw } from '@lucide/svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { serverStore } from '$lib/stores/server.svelte';
	import {
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_CHAT_SECTIONS,
		SETTINGS_SECTION_TITLES,
		type SettingsSection
	} from '$lib/constants';
	import { setMode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums/ui';
	import { fade } from 'svelte/transition';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { setChatSettingsConfigContext } from '$lib/contexts';
	import { settingsReferrer } from '$lib/stores/settings-referrer.svelte';

	interface Props {
		initialSection?: string;
		getSectionHref?: (section: SettingsSection) => string;
	}

	let { initialSection, getSectionHref }: Props = $props();

	let activeSlug = $derived(
		initialSection ?? (page.params as Record<string, string | undefined>).section ?? 'general'
	);
	let currentSection = $derived(
		SETTINGS_CHAT_SECTIONS.find((section) => section.slug === activeSlug) ||
			SETTINGS_CHAT_SECTIONS[0]
	);
	let localConfig: SettingsConfigType = $state({ ...config() });

	let thinkingExpanded = $state(true);
	let localThinkingOverrides = $state<Record<string, number>>({
		...settingsStore.thinkingOverrides
	});

	function paramDisplayName(param: string): string {
		const names: Record<string, string> = {
			temperature: 'Temperature',
			top_p: 'Top P',
			top_k: 'Top K',
			min_p: 'Min P',
			max_tokens: 'Max tokens',
			presence_penalty: 'Presence penalty',
			frequency_penalty: 'Frequency penalty'
		};
		return names[param] ?? param;
	}

	let mobileHeader: { updateCarousel: () => void } | undefined;

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;
		setMode(newTheme as ColorMode);
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig[key] = value;
	}

	function handleReset() {
		localConfig = { ...config() };
		localThinkingOverrides = { ...settingsStore.thinkingOverrides };
		setMode(localConfig.theme as ColorMode);
		mobileHeader?.updateCarousel();
	}

	function handleSave() {
		if (localConfig.custom && typeof localConfig.custom === 'string' && localConfig.custom.trim()) {
			try {
				JSON.parse(localConfig.custom);
			} catch (error) {
				alert('Invalid JSON in custom parameters. Please check the format and try again.');
				console.error(error);
				return;
			}
		}

		const processedConfig = { ...localConfig };

		for (const field of NUMERIC_FIELDS) {
			if (processedConfig[field] !== undefined && processedConfig[field] !== '') {
				const numValue = Number(processedConfig[field]);
				if (!isNaN(numValue)) {
					if ((POSITIVE_INTEGER_FIELDS as readonly string[]).includes(field)) {
						processedConfig[field] = Math.max(1, Math.round(numValue));
					} else {
						processedConfig[field] = numValue;
					}
				} else {
					alert(`Invalid numeric value for ${field}. Please enter a valid number.`);
					return;
				}
			}
		}

		settingsStore.updateMultipleConfig(processedConfig);

		// Persist thinking override changes
		for (const [param, value] of Object.entries(localThinkingOverrides)) {
			const serverOv = serverStore.thinkingOverrides;
			const serverValue = serverOv?.[param];
			if (serverValue !== undefined && value !== serverValue) {
				settingsStore.setThinkingOverride(param, value);
			} else if (serverValue !== undefined && value === serverValue) {
				settingsStore.resetThinkingOverride(param, serverValue);
			}
		}

		goto(settingsReferrer.url);
	}

	export function reset() {
		localConfig = { ...config() };
		localThinkingOverrides = { ...settingsStore.thinkingOverrides };
	}

	setChatSettingsConfigContext({
		get localConfig() {
			return localConfig;
		},
		handleConfigChange,
		handleThemeChange
	});
</script>

<div
	class="mx-auto flex h-full max-h-[100dvh] w-full flex-col overflow-y-auto md:pl-8"
	in:fade={{ duration: 150 }}
>
	<div class="flex flex-1 flex-col gap-4 md:flex-row">
		<SettingsChatDesktopSidebar
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ?? ((section: SettingsSection) => `#/settings/chat/${section.slug}`)}
		/>

		<SettingsChatMobileHeader
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ?? ((section: SettingsSection) => `#/settings/chat/${section.slug}`)}
			bind:this={mobileHeader}
		/>

		<div class="mx-auto max-w-3xl flex-1">
			<div class="space-y-6 p-4 md:p-6 md:pt-28">
				<div class="grid">
					<div class="mb-6 flex items-center gap-2 border-b border-border/30 pb-6 md:flex">
						<currentSection.icon class="h-5 w-5" />
						<h3 class="text-lg font-semibold">{currentSection.title}</h3>
					</div>

					{#if currentSection.title === SETTINGS_SECTION_TITLES.TOOLS}
						<SettingsChatToolsTab />
					{:else if currentSection.title === SETTINGS_SECTION_TITLES.IMPORT_EXPORT}
						<SettingsChatImportExportTab />
					{:else if currentSection.fields}
						<div class="space-y-6">
							<SettingsChatFields
								fields={currentSection.fields}
								{localConfig}
								onConfigChange={handleConfigChange}
								onThemeChange={handleThemeChange}
							/>

							{#if currentSection.title === SETTINGS_SECTION_TITLES.SAMPLING && settingsStore.hasThinkingOverrides}
								<p class="text-xs text-muted-foreground italic">
									When thinking is enabled, the overrides below replace the corresponding
									sampling values.
								</p>

								<div class="border-t border-border/30 pt-4">
									<button
										class="flex w-full items-center gap-2 text-sm font-medium text-foreground"
										onclick={() => (thinkingExpanded = !thinkingExpanded)}
									>
										<Brain class="h-4 w-4" />
										<span>Thinking Mode Overrides</span>
										<ChevronDown
											class="ml-auto h-4 w-4 transition-transform {thinkingExpanded
												? 'rotate-180'
												: ''}"
										/>
									</button>

									{#if thinkingExpanded}
										<div class="mt-3 space-y-3">
											{#each Object.entries(localThinkingOverrides) as [param, value] (param)}
												{@const serverOv = serverStore.thinkingOverrides}
												{@const serverValue = serverOv?.[param]}
												{@const isCustom =
													serverValue !== undefined && value !== serverValue}
												<div class="space-y-1">
													<div class="flex items-center gap-2">
														<label
															for="thinking-{param}"
															class="text-sm font-medium"
														>
															{paramDisplayName(param)}
														</label>
														{#if isCustom}
															<span
																class="rounded bg-accent px-1.5 py-0.5 text-[10px] font-medium text-accent-foreground"
															>
																Custom
															</span>
															<button
																class="inline-flex h-5 w-5 items-center justify-center rounded transition-colors hover:bg-muted"
																title="Reset to server default"
																onclick={() => {
																	if (serverValue !== undefined) {
																		localThinkingOverrides[param] = serverValue;
																	}
																}}
															>
																<RotateCcw class="h-3 w-3" />
															</button>
														{:else}
															<span
																class="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground"
															>
																Default
															</span>
														{/if}
													</div>
													<input
														id="thinking-{param}"
														type="number"
														step="0.01"
														{value}
														onchange={(e) => {
															const v = parseFloat(e.currentTarget.value);
															if (!isNaN(v)) {
																localThinkingOverrides[param] = v;
															}
														}}
														class="h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors md:max-w-md"
													/>
												</div>
											{/each}
										</div>
									{/if}
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<div class="mt-8 border-t border-border/30 pt-6">
					<p class="text-xs text-muted-foreground">Settings are saved in browser's localStorage</p>
				</div>
			</div>

			<SettingsChatFooter onReset={handleReset} onSave={handleSave} />
		</div>
	</div>
</div>
