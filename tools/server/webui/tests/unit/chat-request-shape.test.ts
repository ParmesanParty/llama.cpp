import { describe, expect, it } from 'vitest';

import { buildChatRequest } from '$lib/services/chat-request-builder';

const baseRequestBody = {
	messages: [{ role: 'user', content: 'hi' }],
	stream: true
};

describe('buildChatRequest', () => {
	it('no session, no tools: body has neither session_id nor tools, no token header', () => {
		const { url, init } = buildChatRequest({
			requestBody: { ...baseRequestBody },
			stream: true,
			codeExecSessionId: undefined,
			signal: undefined,
			sessionId: null,
			sessionToken: null
		});
		expect(url).toBe('./v1/chat/completions');
		expect(init.method).toBe('POST');
		const body = JSON.parse(init.body as string);
		expect(body.session_id).toBeUndefined();
		expect(body.tools).toBeUndefined();
		const headers = init.headers as Record<string, string>;
		expect(headers['X-Session-Token']).toBeUndefined();
	});

	it('no session, with tools: body has tools, no session_id, no token header', () => {
		const tools = [
			{
				type: 'function',
				function: { name: '__client.x.echo', description: 'd', parameters: {} }
			}
		];
		const { init } = buildChatRequest({
			requestBody: { ...baseRequestBody, tools },
			stream: true,
			codeExecSessionId: undefined,
			signal: undefined,
			sessionId: null,
			sessionToken: null
		});
		const body = JSON.parse(init.body as string);
		expect(body.tools).toEqual(tools);
		expect(body.session_id).toBeUndefined();
		const headers = init.headers as Record<string, string>;
		expect(headers['X-Session-Token']).toBeUndefined();
	});

	it('session set, with tools: tools is STRIPPED, session_id+token applied', () => {
		const tools = [
			{
				type: 'function',
				function: { name: '__client.x.echo', description: 'd', parameters: {} }
			}
		];
		const { init } = buildChatRequest({
			requestBody: { ...baseRequestBody, tools },
			stream: true,
			codeExecSessionId: undefined,
			signal: undefined,
			sessionId: 'sid-abc',
			sessionToken: 'tok-xyz'
		});
		const body = JSON.parse(init.body as string);
		expect(body.session_id).toBe('sid-abc');
		expect(body.tools).toBeUndefined();
		const headers = init.headers as Record<string, string>;
		expect(headers['X-Session-Token']).toBe('tok-xyz');
	});

	it('session set, no tools: session_id+token applied, no tools field', () => {
		const { init } = buildChatRequest({
			requestBody: { ...baseRequestBody },
			stream: true,
			codeExecSessionId: undefined,
			signal: undefined,
			sessionId: 'sid-abc',
			sessionToken: 'tok-xyz'
		});
		const body = JSON.parse(init.body as string);
		expect(body.session_id).toBe('sid-abc');
		expect(body.tools).toBeUndefined();
		const headers = init.headers as Record<string, string>;
		expect(headers['X-Session-Token']).toBe('tok-xyz');
	});

	it('codeExecSessionId sets X-Code-Exec-Session header', () => {
		const { init } = buildChatRequest({
			requestBody: { ...baseRequestBody },
			stream: false,
			codeExecSessionId: 'conv-42',
			signal: undefined,
			sessionId: null,
			sessionToken: null
		});
		const headers = init.headers as Record<string, string>;
		expect(headers['X-Code-Exec-Session']).toBe('conv-42');
	});

	it('passes through the AbortSignal on init', () => {
		const ctrl = new AbortController();
		const { init } = buildChatRequest({
			requestBody: { ...baseRequestBody },
			stream: true,
			codeExecSessionId: undefined,
			signal: ctrl.signal,
			sessionId: null,
			sessionToken: null
		});
		expect(init.signal).toBe(ctrl.signal);
	});

	it('does not mutate the caller-provided requestBody', () => {
		const tools = [
			{
				type: 'function',
				function: { name: '__client.x.echo', description: 'd', parameters: {} }
			}
		];
		const requestBody = { ...baseRequestBody, tools };
		buildChatRequest({
			requestBody,
			stream: true,
			codeExecSessionId: undefined,
			signal: undefined,
			sessionId: 'sid',
			sessionToken: 'tok'
		});
		expect(requestBody.tools).toEqual(tools);
	});
});
