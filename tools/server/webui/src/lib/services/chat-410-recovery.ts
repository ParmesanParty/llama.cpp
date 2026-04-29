/**
 * Predicate: should the chat request be retried after a 410 response?
 *
 * The proxy returns 410 Gone when it has no record of the session_id in the
 * request body (typically a proxy restart, an idle eviction, or a
 * close-beacon that landed before this request did). The recovery action is
 * to re-register a session via `mergedOrchestrationStore.reconcile()` and
 * retry the request once with the new session.
 *
 * Match against either:
 *   - the structured `code: "session_not_found"` field the proxy sets, or
 *   - the human message "session not found" (case-insensitive) as a fallback
 *     for older proxies or non-JSON error bodies.
 *
 * Other 410s (e.g. permanently retired endpoints) do NOT match — the body
 * has to look like a session-not-found error to trigger retry.
 */
export function shouldRetryAfter410(status: number, bodyText: string): boolean {
	if (status !== 410) return false;
	if (!bodyText) return false;

	try {
		const parsed = JSON.parse(bodyText);
		const code = parsed?.error?.code ?? parsed?.code;
		if (code === 'session_not_found') return true;
		const message = parsed?.error?.message ?? parsed?.message ?? '';
		if (typeof message === 'string' && /session.{0,5}not.{0,5}found/i.test(message)) {
			return true;
		}
		return false;
	} catch {
		// Non-JSON body — fall back to text match.
		return /session.{0,5}not.{0,5}found/i.test(bodyText);
	}
}
