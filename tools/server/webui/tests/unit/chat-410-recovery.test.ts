import { describe, expect, it } from 'vitest';

import { shouldRetryAfter410 } from '$lib/services/chat-410-recovery';

describe('shouldRetryAfter410', () => {
	it('returns false for non-410 statuses', () => {
		expect(shouldRetryAfter410(200, 'whatever')).toBe(false);
		expect(shouldRetryAfter410(400, '{"error":{"code":"session_not_found"}}')).toBe(false);
		expect(shouldRetryAfter410(500, 'session not found')).toBe(false);
	});

	it('returns false for 410 with empty body', () => {
		expect(shouldRetryAfter410(410, '')).toBe(false);
	});

	it('returns true when 410 body is JSON with error.code = session_not_found', () => {
		const body = JSON.stringify({ error: { code: 'session_not_found', message: 'gone' } });
		expect(shouldRetryAfter410(410, body)).toBe(true);
	});

	it('returns true when 410 body is JSON with top-level code = session_not_found', () => {
		expect(shouldRetryAfter410(410, JSON.stringify({ code: 'session_not_found' }))).toBe(true);
	});

	it('returns true when 410 body has a message matching session-not-found (case-insensitive)', () => {
		expect(
			shouldRetryAfter410(410, JSON.stringify({ error: { message: 'Session Not Found' } }))
		).toBe(true);
	});

	it('returns true when 410 body is non-JSON text matching the pattern', () => {
		expect(shouldRetryAfter410(410, 'session not found')).toBe(true);
		expect(shouldRetryAfter410(410, 'SESSION_NOT_FOUND')).toBe(true);
	});

	it('returns false when 410 body is JSON with unrelated error code', () => {
		expect(
			shouldRetryAfter410(
				410,
				JSON.stringify({ error: { code: 'gone_forever', message: 'this URL is retired' } })
			)
		).toBe(false);
	});

	it('returns false when 410 body is non-JSON unrelated text', () => {
		expect(shouldRetryAfter410(410, 'endpoint retired')).toBe(false);
	});
});
