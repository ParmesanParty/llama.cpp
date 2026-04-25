import { describe, test, expect, vi, beforeEach } from 'vitest';
import { AttachmentType } from '$lib/enums';
import { createStreamEventHandlers } from '$lib/stores/stream-event-handlers';
import type { StreamEvent } from '$lib/types';

beforeEach(() => {
	// pushEvent schedules a flush via requestAnimationFrame; the node test
	// environment doesn't define it. Stub with a no-op so handlers run cleanly.
	vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
		// Return a fake handle; we don't run the callback in unit tests because
		// these tests assert on synchronous side-effects of the handlers.
		void cb;
		return 1;
	});
});

function makeCtx() {
	const onAttachments = vi.fn();
	const ctx = {
		streamedEvents: [] as StreamEvent[],
		contentOffset: () => 0,
		finalizeReasoning: vi.fn(),
		updateStreamingContent: vi.fn(),
		findMessageIndex: vi.fn(() => 0),
		updateMessageAtIndex: vi.fn(),
		assistantMessageId: 'assistant-msg-id',
		sentMessages: () => [],
		activeMessages: () => [],
		activeConversation: () => null,
		setCompaction: vi.fn(),
		onAttachments
	};
	return { ctx, onAttachments };
}

describe('createStreamEventHandlers.onToolArtifacts', () => {
	test('image artifact promotes to onAttachments as IMAGE extra', () => {
		const { ctx, onAttachments } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const handlers = createStreamEventHandlers(ctx as any);

		handlers.onToolArtifacts({
			tool: 'code_exec',
			call_id: 'c1',
			iteration: 1,
			artifact: {
				kind: 'image',
				name: 'plot.png',
				mime: 'image/png',
				data_b64: 'QUJD',
				width: 640,
				height: 480,
				url: '/images/plot.png'
			}
		});

		expect(onAttachments).toHaveBeenCalledOnce();
		const [msgId, extras] = onAttachments.mock.calls[0];
		expect(msgId).toBe('assistant-msg-id');
		expect(extras).toHaveLength(1);
		expect(extras[0]).toEqual({
			type: AttachmentType.IMAGE,
			name: 'plot.png',
			base64Url: 'data:image/png;base64,QUJD',
			width: 640,
			height: 480,
			url: '/images/plot.png'
		});
	});

	test('image artifact without dimensions omits width/height keys', () => {
		const { ctx, onAttachments } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const handlers = createStreamEventHandlers(ctx as any);
		handlers.onToolArtifacts({
			tool: 'code_exec',
			call_id: 'c1',
			iteration: 1,
			artifact: { kind: 'image', name: 'x.png', mime: 'image/png', data_b64: 'Xx==' }
		});
		const [, extras] = onAttachments.mock.calls[0];
		expect(extras[0]).not.toHaveProperty('width');
		expect(extras[0]).not.toHaveProperty('height');
		expect(extras[0]).not.toHaveProperty('url');
		expect(extras[0].base64Url).toBe('data:image/png;base64,Xx==');
	});

	test('non-image artifact kind does not call onAttachments but still pushEvent', () => {
		const { ctx, onAttachments } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const handlers = createStreamEventHandlers(ctx as any);
		handlers.onToolArtifacts({
			tool: 'some_future_tool',
			call_id: 'c2',
			iteration: 1,
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			artifact: { kind: 'unknown_future_kind' } as any
		});
		expect(onAttachments).not.toHaveBeenCalled();
		// pushEvent still fires for stream-event history rendering
		expect(ctx.streamedEvents).toHaveLength(1);
	});

	test('missing onAttachments callback is silent no-op', () => {
		const { ctx } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		delete (ctx as any).onAttachments;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const handlers = createStreamEventHandlers(ctx as any);
		expect(() =>
			handlers.onToolArtifacts({
				tool: 'code_exec',
				call_id: 'c1',
				iteration: 1,
				artifact: { kind: 'image', name: 'x.png', mime: 'image/png', data_b64: 'AA==' }
			})
		).not.toThrow();
		expect(ctx.streamedEvents).toHaveLength(1);
	});
});
