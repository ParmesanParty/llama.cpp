import { toast } from 'svelte-sonner';
import { modelsStore } from '$lib/stores/models.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { toolHealthStore } from '$lib/stores/toolHealth.svelte';

/**
 * EventsService — SSE client for proxy push notifications.
 *
 * Connects to GET /api/events and dispatches events to stores.
 * EventSource auto-reconnects natively (~3s default backoff).
 * On every reconnect, re-fetches server state to reconcile.
 */
export class EventsService {
	private static eventSource: EventSource | null = null;
	private static reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	private static hasConnected = false;

	/**
	 * Connect to the SSE event stream.
	 * Safe to call multiple times — no-ops if already connected.
	 */
	static connect(): void {
		if (this.eventSource) return;

		this.eventSource = new EventSource('/api/events');

		this.eventSource.onopen = () => {
			// Reconcile on every connect/reconnect — presets may have changed
			// while disconnected.  Layout also fires an eager fetchPresets()
			// on mount so the model switcher doesn't wait for SSE.
			modelsStore.fetchPresets();
			// Skip redundant props fetch on initial connect — layout handles it.
			if (this.hasConnected) {
				serverStore.fetch();
			}
			// Reconcile tool-health on every connect/reconnect — breakers may
			// have transitioned while we were disconnected.
			toolHealthStore.fetchSnapshot();
			toolHealthStore.markSseConnected();
			this.hasConnected = true;
		};

		this.eventSource.addEventListener('model-switch-started', (e: MessageEvent) => {
			try {
				const data = JSON.parse(e.data);
				modelsStore.switching = true;
				modelsStore.switchError = null;
				console.info('[Events] Model switch started:', data.model);
			} catch {
				// Ignore malformed events
			}
		});

		this.eventSource.addEventListener('model-switch-completed', (e: MessageEvent) => {
			try {
				const data = JSON.parse(e.data);
				modelsStore.clearSwitchTimeout();
				modelsStore.switching = false;
				modelsStore.fetchPresets();
				serverStore.fetch();
				console.info('[Events] Model switch completed:', data.model);
			} catch {
				// Ignore malformed events
			}
		});

		this.eventSource.addEventListener('model-switch-failed', (e: MessageEvent) => {
			try {
				const data = JSON.parse(e.data);
				modelsStore.clearSwitchTimeout();
				modelsStore.switching = false;
				modelsStore.switchError = data.error || 'Switch failed';
				toast.error(`Model switch failed: ${data.error || 'Unknown error'}`);
				console.warn('[Events] Model switch failed:', data.error);
			} catch {
				// Ignore malformed events
			}
		});

		this.eventSource.addEventListener('tool-health-changed', (e: MessageEvent) => {
			try {
				const data = JSON.parse(e.data);
				toolHealthStore.applyChange(data);
			} catch {
				// Ignore malformed events
			}
		});

		this.eventSource.onerror = () => {
			// EventSource auto-reconnects on transient errors.
			// If the connection closes permanently, fall back to manual reconnect.
			if (this.eventSource?.readyState === EventSource.CLOSED) {
				toolHealthStore.markSseDisconnected();
				this.disconnect();
				this.reconnectTimer = setTimeout(() => {
					this.reconnectTimer = null;
					this.connect();
				}, 5000);
			}
		};
	}

	/**
	 * Disconnect from the SSE event stream.
	 */
	static disconnect(): void {
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		if (this.eventSource) {
			this.eventSource.close();
			this.eventSource = null;
		}
		this.hasConnected = false;
		toolHealthStore.markSseDisconnected();
	}
}
