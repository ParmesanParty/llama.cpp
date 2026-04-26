import { base } from '$app/paths';

import { mcpStore } from './mcp.svelte';

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

class MergedOrchestrationStore {
	private _capability = $state<ProxyCapability | null>(null);
	private _session = $state<SessionInfo | null>(null);

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

		const clientTools = this._buildClientToolsList();
		const serverEnablement: Record<string, boolean> = {};

		try {
			const resp = await fetch(`${base}${this._capability.endpoints.sessions}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					client_tools: clientTools,
					server_tool_enablement: serverEnablement
				})
			});
			if (!resp.ok) {
				console.warn('[MergedOrchestration] session registration failed', resp.status);
				return;
			}
			const body = await resp.json();
			this._session = {
				sessionId: body.session_id,
				sessionToken: body.session_token
			};
			sessionStorage.setItem('mo.sessionId', body.session_id);
			sessionStorage.setItem('mo.sessionToken', body.session_token);
		} catch (e) {
			console.warn('[MergedOrchestration] session registration threw', e);
		}
	}

	closeSession(): void {
		if (!this._session) return;
		try {
			navigator.sendBeacon(
				`${base}/api/sessions/${this._session.sessionId}/close`,
				new Blob([JSON.stringify({})], { type: 'application/json' })
			);
		} catch {
			// best effort; session will idle-expire on the server
		}
		this._session = null;
		sessionStorage.removeItem('mo.sessionId');
		sessionStorage.removeItem('mo.sessionToken');
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
