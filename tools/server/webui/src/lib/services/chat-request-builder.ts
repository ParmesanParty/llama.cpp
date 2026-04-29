import { getJsonHeaders, getStreamHeaders } from '$lib/utils/api-headers';

export interface BuildChatRequestParams {
	requestBody: Record<string, unknown>;
	stream: boolean;
	/** If set, X-Code-Exec-Session header pins the kernel pool entry so
	 * variables/imports persist across code_exec calls in the same conversation. */
	codeExecSessionId?: string;
	signal?: AbortSignal;
	/** Merged-orchestration session id (null if not registered). */
	sessionId: string | null;
	/** Merged-orchestration session token (null if not registered). */
	sessionToken: string | null;
}

/**
 * Build the URL + RequestInit for a /v1/chat/completions POST.
 *
 * Contract:
 * - When a merged-orchestration session is registered (sessionId+sessionToken
 *   both present), `body.session_id` is set, the `X-Session-Token` header is
 *   added, AND `body.tools` is removed. The proxy's session catalog already
 *   knows the tool list, and sending both `tools` and `session_id` triggers
 *   the proxy's `incoherent_request` 400 reject.
 * - When no session is registered, the body passes through unchanged.
 * - The caller-provided `requestBody` is never mutated.
 */
export function buildChatRequest(params: BuildChatRequestParams): {
	url: string;
	init: RequestInit;
} {
	const { requestBody, stream, codeExecSessionId, signal, sessionId, sessionToken } = params;

	const headers: Record<string, string> = stream ? getStreamHeaders() : getJsonHeaders();
	if (codeExecSessionId) {
		headers['X-Code-Exec-Session'] = codeExecSessionId;
	}

	const body: Record<string, unknown> = { ...requestBody };
	if (sessionId && sessionToken) {
		body.session_id = sessionId;
		headers['X-Session-Token'] = sessionToken;
		// Server runs the unified tool loop via the session catalog. Drop any
		// caller-supplied `tools` to avoid the proxy's incoherent_request 400 —
		// and to actually make merged-orchestration take effect.
		delete body.tools;
	} else {
		delete body.session_id;
	}

	return {
		url: `./v1/chat/completions`,
		init: { method: 'POST', headers, body: JSON.stringify(body), signal }
	};
}
