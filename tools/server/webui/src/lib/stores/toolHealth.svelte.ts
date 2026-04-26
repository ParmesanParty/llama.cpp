/**
 * Tool health store — backs the Tools sidebar entry, badge dot, and modal.
 *
 * Populated eagerly at app init via fetchSnapshot(). Updated incrementally
 * by the /api/events subscriber when tool-health-changed events fire.
 *
 * The store is module-scoped so any component can import { toolHealthStore }
 * and read $state runes directly. SSE wiring lives in events.service.ts;
 * the snapshot fetch is also kicked off from the layout's onMount alongside
 * EventsService.connect().
 */

export type BreakerState = 'closed' | 'open' | 'half_open';

export interface ToolHealthEntry {
	name: string;
	display_name: string;
	summary: string;
	parameters: Record<string, unknown>;
	breaker_state: BreakerState;
	recent_failures: number;
	last_failure: number | null;
	last_success: number | null;
	preflight_ok: boolean;
	preflight_reason: string | null;
}

export interface ServerStateSnapshot {
	state: string;
	uptime_seconds: number;
	active_requests: number;
}

export type BadgeColor = 'gray' | 'green' | 'amber' | 'red';

// Three-state SSE lifecycle. 'pending' is the boot-time state before any
// onopen has fired — distinct from 'disconnected' so the badge can render
// gray (status unknown) instead of red (live updates lost) at cold start.
export type SseState = 'pending' | 'connected' | 'disconnected';

interface HealthApiResponse {
	tools: Record<string, Omit<ToolHealthEntry, 'name'>>;
	server: ServerStateSnapshot;
}

interface ToolHealthChangedPayload {
	tool: string;
	breaker_state: BreakerState;
	recent_failures: number;
	last_failure: number | null;
	last_success: number | null;
}

class ToolHealthStore {
	tools: Map<string, ToolHealthEntry> = $state(new Map());
	server: ServerStateSnapshot | null = $state(null);
	loading = $state(false);
	error: string | null = $state(null);
	// 'connected' between SSE onopen and the next permanent onerror+CLOSED.
	// 'pending' before the first connect (boot-time gray window). 'disconnected'
	// only after a confirmed permanent close — transient errors don't flip it,
	// EventSource auto-reconnects, and EventsService only calls
	// markSseDisconnected on the permanent CLOSED path.
	sseState: SseState = $state('pending');

	badgeColor = $derived.by<BadgeColor>(() => {
		if (this.error) return 'red';
		if (this.tools.size === 0) {
			// No snapshot yet: gray during the boot window, red only once we've
			// seen SSE drop (which implies the snapshot will keep failing too).
			return this.sseState === 'disconnected' ? 'red' : 'gray';
		}
		// We have data — treat live-update loss as a hard error so the badge
		// reflects that the displayed state may be stale.
		if (this.sseState === 'disconnected') return 'red';
		let degraded = 0;
		for (const t of this.tools.values()) {
			if (t.breaker_state !== 'closed' || !t.preflight_ok) degraded++;
		}
		if (degraded === 0) return 'green';
		return 'amber';
	});

	badgeTooltip = $derived.by<string>(() => {
		if (this.error) return 'Tool service unreachable';
		if (this.tools.size === 0) {
			if (this.sseState === 'disconnected') return 'Tool service unreachable';
			return this.loading ? 'Loading tool status…' : 'Tool status unknown';
		}
		if (this.sseState === 'disconnected') return 'Tool service unreachable';
		const total = this.tools.size;
		let degraded = 0;
		for (const t of this.tools.values()) {
			if (t.breaker_state !== 'closed' || !t.preflight_ok) degraded++;
		}
		if (degraded === 0) return `All ${total} tools healthy`;
		return `${degraded} of ${total} tools degraded`;
	});

	async fetchSnapshot(): Promise<void> {
		this.loading = true;
		this.error = null;
		try {
			const resp = await fetch('/api/tool-health');
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const body = (await resp.json()) as HealthApiResponse;
			const next = new Map<string, ToolHealthEntry>();
			for (const [name, entry] of Object.entries(body.tools)) {
				next.set(name, { name, ...entry });
			}
			this.tools = next;
			this.server = body.server;
		} catch (err) {
			this.error = err instanceof Error ? err.message : String(err);
		} finally {
			this.loading = false;
		}
	}

	applyChange(payload: ToolHealthChangedPayload): void {
		const existing = this.tools.get(payload.tool);
		if (!existing) return; // unknown tool — ignore until next snapshot
		const next = new Map(this.tools);
		next.set(payload.tool, {
			...existing,
			breaker_state: payload.breaker_state,
			recent_failures: payload.recent_failures,
			last_failure: payload.last_failure,
			last_success: payload.last_success,
		});
		this.tools = next;
	}

	async resetBreaker(toolName: string): Promise<void> {
		const resp = await fetch(`/api/tool-health/${encodeURIComponent(toolName)}/reset`, {
			method: 'POST',
		});
		if (!resp.ok) {
			throw new Error(`Reset failed: HTTP ${resp.status}`);
		}
		// State update arrives via the tool-health-changed SSE event.
	}

	markSseConnected(): void {
		this.sseState = 'connected';
	}

	markSseDisconnected(): void {
		this.sseState = 'disconnected';
	}
}

export const toolHealthStore = new ToolHealthStore();
