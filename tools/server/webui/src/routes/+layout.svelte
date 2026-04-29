<script lang="ts">
	import '../app.css';
	import { base } from '$app/paths';
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { untrack } from 'svelte';
	import { onMount } from 'svelte';
	import { fade } from 'svelte/transition';
	import {
		DesktopIconStrip,
		DialogConversationTitleUpdate,
		SidebarNavigation
	} from '$lib/components/app';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import * as Sidebar from '$lib/components/ui/sidebar/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { EventsService } from '$lib/services';
	import { isRouterMode, serverStore } from '$lib/stores/server.svelte';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import { toolHealthStore } from '$lib/stores/toolHealth.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import { Toaster } from 'svelte-sonner';
	import { modelsStore } from '$lib/stores/models.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mergedOrchestrationStore } from '$lib/stores/merged-orchestration.svelte';
	import { TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { useKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
	import { useSettingsNavigation } from '$lib/hooks/use-settings-navigation.svelte';
	import { conversations } from '$lib/stores/conversations.svelte';

	let { children } = $props();

	let alwaysShowSidebarOnDesktop = $derived(config().alwaysShowSidebarOnDesktop);
	let isMobile = new IsMobile();
	let isDesktop = $derived(!isMobile.current);
	let sidebarOpen = $state(false);
	let mounted = $state(false);
	let innerHeight = $state<number | undefined>();
	let chatSidebar:
		| { activateSearchMode?: () => void; editActiveConversation?: () => void }
		| undefined = $state();

	let titleUpdateDialogOpen = $state(false);
	let titleUpdateCurrentTitle = $state('');
	let titleUpdateNewTitle = $state('');
	let titleUpdateResolve: ((value: boolean) => void) | null = null;

	const panelNav = useSettingsNavigation();

	function navigateToConversation(direction: -1 | 1) {
		const allConvs = conversations();
		if (allConvs.length === 0) return;

		const currentId = page.params.id;

		if (!currentId) {
			goto(`#/chat/${allConvs[direction === 1 ? 0 : allConvs.length - 1].id}`);

			return;
		}

		const idx = allConvs.findIndex((c) => c.id === currentId);
		if (idx === -1) return;

		const targetIdx = idx + direction;

		if (targetIdx >= 0 && targetIdx < allConvs.length) {
			goto(`#/chat/${allConvs[targetIdx].id}`);
		} else {
			goto('?new_chat=true#/');
		}
	}

	// Global keyboard shortcuts
	const { handleKeydown } = useKeyboardShortcuts({
		editActiveConversation: () => chatSidebar?.editActiveConversation?.(),

		navigateToPrevConversation: () => navigateToConversation(-1),

		navigateToNextConversation: () => navigateToConversation(1)
	});

	function checkApiKey() {
		const apiKey = config().apiKey;

		if (
			(page.route.id === '/(chat)' || page.route.id === '/(chat)/chat/[id]') &&
			page.status !== 401 &&
			page.status !== 403
		) {
			const headers: Record<string, string> = {
				'Content-Type': 'application/json'
			};

			if (apiKey && apiKey.trim() !== '') {
				headers.Authorization = `Bearer ${apiKey.trim()}`;
			}

			fetch(`${base}/props`, { headers })
				.then((response) => {
					if (response.status === 401 || response.status === 403) {
						window.location.reload();
					}
				})
				.catch((e) => {
					console.error('Error checking API key:', e);
				});
		}
	}

	function handleTitleUpdateCancel() {
		titleUpdateDialogOpen = false;

		if (titleUpdateResolve) {
			titleUpdateResolve(false);
			titleUpdateResolve = null;
		}
	}

	function handleTitleUpdateConfirm() {
		titleUpdateDialogOpen = false;

		if (titleUpdateResolve) {
			titleUpdateResolve(true);
			titleUpdateResolve = null;
		}
	}

	onMount(() => {
		mounted = true;
	});

	$effect(() => {
		if (alwaysShowSidebarOnDesktop && isDesktop) {
			sidebarOpen = true;
			return;
		}
	});

	// Initialize server properties on app load (run once)
	$effect(() => {
		// Only fetch if we don't already have props
		if (!serverStore.props) {
			untrack(() => {
				serverStore.fetch();
			});
		}
	});

	// Apply preserve-thinking default once server props are known.
	// clearActiveConversation() runs on the index route's onMount before /props
	// has resolved, so the initial default would otherwise be `false`. Re-apply
	// when the supported flag changes — only while no chat is active so we
	// don't clobber a loaded conversation's stored value.
	$effect(() => {
		const supported = serverStore.preserveThinkingSupported;
		untrack(() => {
			if (!conversationsStore.activeConversation) {
				conversationsStore.activePreserveThinking = supported;
			}
		});
	});

	// [parmesan] Eager preset fetch — don't wait for SSE onopen to discover
	// switchable models.  EventsService.onopen still re-fetches on reconnect
	// to reconcile after disconnects.
	$effect(() => {
		if (browser) {
			untrack(() => {
				modelsStore.fetchPresets();
			});
		}
	});

	// Connect to SSE events (onopen handler re-fetches presets on reconnect)
	$effect(() => {
		if (browser) {
			untrack(() => {
				toolHealthStore.fetchSnapshot();
				EventsService.connect();
			});
			return () => EventsService.disconnect();
		}
	});


	// Sync settings when server props are loaded
	$effect(() => {
		const serverProps = serverStore.props;

		if (serverProps) {
			untrack(() => {
				settingsStore.syncWithServerDefaults();
			});
		}
	});

	// Fetch router models when in router mode (for status and modalities)
	// Wait for models to be loaded first, run only once
	let routerModelsFetched = false;

	$effect(() => {
		const isRouter = isRouterMode();
		const modelsCount = modelsStore.models.length;

		// Only fetch router models once when we have models loaded and in router mode
		if (isRouter && modelsCount > 0 && !routerModelsFetched) {
			routerModelsFetched = true;
			untrack(() => {
				modelsStore.fetchRouterModels();
			});
		}
	});

	// Background MCP server health checks on app load
	// Fetch enabled servers from settings and run health checks in background
	$effect(() => {
		if (!browser) return;

		const mcpServers = mcpStore.getServers();

		// Only run health checks if we have enabled servers with URLs
		const enabledServers = mcpServers.filter((s) => s.enabled && s.url.trim());

		if (enabledServers.length > 0) {
			untrack(() => {
				// Run health checks in background (don't await)
				mcpStore.runHealthChecksForServers(enabledServers, false).catch((error) => {
					console.warn('[layout] MCP health checks failed:', error);
				});
			});
		}
	});

	// Monitor API key changes and redirect to error page if removed or changed when required
	$effect(() => {
		checkApiKey();
	});

	// Merged orchestration: probe capability + register session if available.
	$effect(() => {
		if (!browser) return;
		untrack(() => {
			(async () => {
				await mergedOrchestrationStore.probeCapability();
				if (mergedOrchestrationStore.isEnabled) {
					await mergedOrchestrationStore.registerSession();
				}
			})();
		});
	});

	// Best-effort session close on tab hide / unload.
	$effect(() => {
		if (!browser) return;
		const handler = () => mergedOrchestrationStore.closeSession();
		window.addEventListener('pagehide', handler);
		return () => window.removeEventListener('pagehide', handler);
	});

	// Merged orchestration: reconcile session when MCP server set changes.
	let lastReconcileSignature = '';
	$effect(() => {
		if (!browser) return;
		const sig = JSON.stringify([...mcpStore.connectedServerNames].sort());
		if (sig === lastReconcileSignature) return;
		const previous = lastReconcileSignature;
		lastReconcileSignature = sig;

		// Skip the very first run: the probe-and-register effect above handles
		// initial registration with the correct connected-server set.
		if (previous === '') return;
		if (!mergedOrchestrationStore.isEnabled) return;
		if (!mergedOrchestrationStore.sessionId) return;

		untrack(() => {
			(async () => {
				try {
					await mergedOrchestrationStore.reconcile();
				} catch (e) {
					console.warn('[layout] merged orchestration reconcile failed', e);
				}
			})();
		});
	});

	// Set up title update confirmation callback
	$effect(() => {
		conversationsStore.setTitleUpdateConfirmationCallback(
			async (currentTitle: string, newTitle: string) => {
				return new Promise<boolean>((resolve) => {
					titleUpdateCurrentTitle = currentTitle;
					titleUpdateNewTitle = newTitle;
					titleUpdateResolve = resolve;
					titleUpdateDialogOpen = true;
				});
			}
		);
	});
</script>

<Tooltip.Provider delayDuration={TOOLTIP_DELAY_DURATION}>
	<ModeWatcher />

	<Toaster richColors />

	<DialogConversationTitleUpdate
		bind:open={titleUpdateDialogOpen}
		currentTitle={titleUpdateCurrentTitle}
		newTitle={titleUpdateNewTitle}
		onConfirm={handleTitleUpdateConfirm}
		onCancel={handleTitleUpdateCancel}
	/>

	<Sidebar.Provider bind:open={sidebarOpen}>
		<div class="flex h-screen w-full" style:height="{innerHeight}px">
			<Sidebar.Root variant="floating" class="h-full">
				<SidebarNavigation bind:this={chatSidebar} />
			</Sidebar.Root>

			{#if !(alwaysShowSidebarOnDesktop && isDesktop) && !(panelNav.isSettingsRoute && !isDesktop)}
				{#if mounted}
					<div in:fade={{ duration: 200 }}>
						<Sidebar.Trigger
							class="transition-left absolute left-0 z-[900] duration-200 ease-linear {sidebarOpen
								? 'left-[calc(var(--sidebar-width)+0.75rem)] max-md:hidden'
								: 'left-0!'}"
							style="translate: 1rem 1rem;"
						/>
					</div>
				{/if}
			{/if}

			{#if isDesktop && !alwaysShowSidebarOnDesktop}
				<DesktopIconStrip
					{sidebarOpen}
					onSearchClick={() => {
						if (chatSidebar?.activateSearchMode) {
							chatSidebar.activateSearchMode();
						}

						sidebarOpen = true;
					}}
				/>
			{/if}

			<Sidebar.Inset class="flex flex-1 flex-col overflow-hidden">
				{@render children?.()}
			</Sidebar.Inset>
		</div>
	</Sidebar.Provider>
</Tooltip.Provider>

<svelte:window onkeydown={handleKeydown} bind:innerHeight />
