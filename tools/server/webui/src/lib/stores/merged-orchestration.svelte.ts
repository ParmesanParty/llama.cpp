import { base } from '$app/paths';

import { mcpStore } from './mcp.svelte';
import { toolsStore } from './tools.svelte';

interface SessionInfo {
	sessionId: string;
	sessionToken: string;
}

interface ProxyCapability {
	enabled: boolean;
	version: number;
	endpoints: { sessions: string; tool_callback: string };
	limits: Record<string, number>;
}

export class MergedOrchestrationStore {
	private _capability = $state<ProxyCapability | null>(null);
	private _session = $state<SessionInfo | null>(null);
	private _registrationInFlight: Promise<SessionInfo | null> | null = null;

	get isEnabled(): boolean {
		return this._capability?.enabled === true;
	}

	get sessionId(): string | null {
		return this._session?.sessionId ?? null;
	}

	get sessionToken(): string | null {
		return this._session?.sessionToken ?? null;
	}

	async probeCapability(): Promise<void> {
		try {
			const resp = await fetch(`${base}/api/proxy-status`);
			if (!resp.ok) return;
			const body = await resp.json();
			const mo = body.merged_orchestration;
			if (mo?.enabled) {
				this._capability = mo as ProxyCapability;
			}
		} catch (e) {
			console.warn('[MergedOrchestration] capability probe failed', e);
		}
	}

	async registerSession(): Promise<void> {
		if (!this._capability?.enabled) return;
		if (this._session) return;

		const next = await this._registerNewSession();
		if (next) {
			this._session = next;
		}
	}

	/**
	 * Re-register the session against the proxy so client_tools reflect the
	 * current MCP state. Atomic-swap: register the new session first, then
	 * replace, then close-beacon the old session out-of-band. Any reader of
	 * sessionId/sessionToken during the in-flight registration sees the OLD
	 * session — slightly-stale-but-valid is strictly better than null.
	 *
	 * Returns true iff the swap occurred. Failure preserves the old session;
	 * the 410-recovery path treats failure as "old is dead" and explicitly
	 * nulls via closeSession(); the MCP-set-change trigger leaves it in
	 * place (the proxy's catalog is only slightly stale).
	 */
	async reconcile(): Promise<boolean> {
		if (!this._capability?.enabled) return false;

		const next = await this._registerNewSession();
		if (!next) return false;

		const previous = this._session;
		this._session = next;

		// Reference-equality guard: when registerSession (cold-start) and
		// reconcile coalesce on the same in-flight registration, both resolve
		// with the SAME SessionInfo object. registerSession's caller assigns
		// _session = next first; reconcile would then beacon the just-minted
		// session. Skip the beacon — the swap is a no-op.
		if (previous && previous !== next) {
			this._sendCloseBeacon(previous.sessionId, previous.sessionToken);
		}
		return true;
	}

	closeSession(): void {
		if (!this._session) return;
		const { sessionId, sessionToken } = this._session;
		this._sendCloseBeacon(sessionId, sessionToken);
		this._session = null;
	}

	/**
	 * PATCH session.server_tool_enablement when the user toggles a builtin
	 * tool in ChatSettingsToolsTab. The proxy stores the per-session map and
	 * filters auto-injection inside `build_merged_catalog` on every iteration
	 * — no per-request header needed on the merged-orch path.
	 *
	 * The proxy's PATCH semantics are "set these keys" (sessions.py:2344
	 * iterates and assigns `s.server_tool_enablement[k] = v`), NOT replace
	 * the full map — so to flip a tool from disabled back to enabled we MUST
	 * include that tool with `true` in the body. Caller passes the full
	 * builtin-tool name list so we can emit the complete enabled/disabled
	 * state rather than only the disabled subset.
	 *
	 * Best-effort: a failed PATCH leaves the proxy's view stale until the
	 * next reconcile; toggling is interactive so the user can re-toggle if
	 * a request shows the old behavior.
	 */
	async applyServerToolEnablement(
		allBuiltinNames: string[],
		disabledBuiltinTools: string[]
	): Promise<void> {
		if (!this._session) return;
		const { sessionId, sessionToken } = this._session;
		const disabled = new Set(disabledBuiltinTools);
		const setMap: Record<string, boolean> = {};
		for (const name of allBuiltinNames) {
			setMap[name] = !disabled.has(name);
		}
		try {
			const resp = await fetch(`${base}/api/sessions/${sessionId}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json', 'X-Session-Token': sessionToken },
				body: JSON.stringify({ set_server_tool_enablement: setMap })
			});
			if (!resp.ok) {
				console.warn('[MergedOrchestration] PATCH server_tool_enablement failed', resp.status);
			}
		} catch (e) {
			console.warn('[MergedOrchestration] PATCH server_tool_enablement threw', e);
		}
	}

	/** Build the initial server_tool_enablement map sent on session register.
	 * Disabled builtins map to `false`; enabled tools are omitted (the proxy
	 * defaults missing keys to enabled). Sufficient because session creation
	 * starts from an empty map — PATCH later carries the full state when
	 * re-enabling matters. */
	private _buildServerEnablement(): Record<string, boolean> {
		const map: Record<string, boolean> = {};
		for (const name of toolsStore.disabledBuiltinTools) map[name] = false;
		return map;
	}

	/**
	 * Resolves when no session registration is in flight. Callers (e.g.
	 * chat.service.sendMessage) await this before reading sessionId so they
	 * don't snapshot a transient null during cold-start ($effect-fire to
	 * fetch-resolve) or during a reconcile in progress.
	 */
	async waitUntilReady(): Promise<void> {
		if (this._registrationInFlight) {
			await this._registrationInFlight;
		}
	}

	private async _registerNewSession(): Promise<SessionInfo | null> {
		// Coalesce concurrent calls. registerSession (cold-start) and reconcile
		// (MCP change) can fire near-simultaneously; one HTTP round-trip and
		// one in-flight promise for waitUntilReady() to await.
		if (this._registrationInFlight) {
			return this._registrationInFlight;
		}

		const promise = (async (): Promise<SessionInfo | null> => {
			const clientTools = this._buildClientToolsList();
			const serverEnablement = this._buildServerEnablement();

			try {
				const resp = await fetch(`${base}${this._capability!.endpoints.sessions}`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						client_tools: clientTools,
						server_tool_enablement: serverEnablement
					})
				});
				if (!resp.ok) {
					console.warn('[MergedOrchestration] session registration failed', resp.status);
					return null;
				}
				const body = await resp.json();
				return {
					sessionId: body.session_id,
					sessionToken: body.session_token
				};
			} catch (e) {
				console.warn('[MergedOrchestration] session registration threw', e);
				return null;
			}
		})();

		this._registrationInFlight = promise;
		try {
			return await promise;
		} finally {
			this._registrationInFlight = null;
		}
	}

	private _sendCloseBeacon(sessionId: string, sessionToken: string): void {
		try {
			// sendBeacon cannot set custom headers; the proxy accepts ?token= on
			// this endpoint. Token is per-session and short-lived; URL-borne
			// auth is acceptable here (see proxy.py:_extract_session_token).
			navigator.sendBeacon(
				`${base}/api/sessions/${sessionId}/close?token=${encodeURIComponent(sessionToken)}`,
				new Blob([JSON.stringify({})], { type: 'application/json' })
			);
		} catch {
			// best effort; session will idle-expire on the server
		}
	}

	private _buildClientToolsList(): unknown[] {
		const defs = mcpStore.getToolDefinitionsForLLM();
		const result: unknown[] = [];
		for (const d of defs) {
			const local = d.function.name;
			const alias = mcpStore.getToolServer(local) ?? 'unknown';
			result.push({
				name: `__client.${alias}.${local}`,
				description: d.function.description,
				parameters: d.function.parameters,
				server_alias: alias,
				local_name: local,
				sanitize_outbound: false,
				sanitize_inbound_html: false,
				sanitize_inbound_injection: true
			});
		}
		return result;
	}
}

export const mergedOrchestrationStore = new MergedOrchestrationStore();
