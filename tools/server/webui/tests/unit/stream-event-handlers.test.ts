import { describe, test, expect, vi, beforeEach } from 'vitest';
import { AttachmentType } from '$lib/enums';
import { createStreamEventHandlers } from '$lib/stores/stream-event-handlers';
import type { StreamEvent } from '$lib/types';

beforeEach(() => {
	// pushEvent schedules a flush via requestAnimationFrame; the node test
	// environment doesn't define it. Run the callback synchronously so
	// flushEvents (which writes streamedEvents back to the message via
	// updateMessageAtIndex) is actually exercised in tests.
	vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
		cb(0);
		return 1;
	});
});

function makeCtx() {
	const onAttachments = vi.fn();
	const ctx = {
		streamedEvents: [] as StreamEvent[],
		contentOffset: () => 42,
		finalizeReasoning: vi.fn(),
		updateStreamingContent: vi.fn(),
		findMessageIndex: vi.fn(() => 0),
		updateMessageAtIndex: vi.fn(),
		getCurrentMessageId: () => 'assistant-msg-id',
		sentMessages: () => [] as never,
		activeMessages: () => [] as never,
		activeConversation: (() => null) as () => { id: string } | null,
		setCompaction: vi.fn(),
		onAttachments
	};
	return { ctx, onAttachments };
}

describe('createStreamEventHandlers.onToolStatus', () => {
	test('executing status finalizes reasoning and flushes content', () => {
		const { ctx } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onToolStatus({ tool: 'web_search', status: 'executing', iteration: 1 });
		expect(ctx.finalizeReasoning).toHaveBeenCalledOnce();
		expect(ctx.updateStreamingContent).toHaveBeenCalledOnce();
		expect(ctx.streamedEvents).toHaveLength(1);
		expect(ctx.streamedEvents[0]).toMatchObject({ type: 'tool_status', offset: 42 });
	});

	test('non-executing status pushes event without flushing reasoning', () => {
		const { ctx } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onToolStatus({ tool: 'web_search', status: 'completed', iteration: 1 });
		expect(ctx.finalizeReasoning).not.toHaveBeenCalled();
		expect(ctx.streamedEvents).toHaveLength(1);
	});
});

describe('createStreamEventHandlers.onSources', () => {
	test('sources event always lands at offset 0 (sticky-bottom footer)', () => {
		const { ctx } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onSources({ sources: [{ index: 0, title: 't', url: 'u' }] });
		expect(ctx.streamedEvents[0]).toMatchObject({ type: 'sources', offset: 0 });
	});
});

describe('createStreamEventHandlers flush behavior', () => {
	test('flushEvents writes a snapshot copy (not a live reference) to the message', () => {
		const { ctx } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onToolStatus({ tool: 'web_search', status: 'completed', iteration: 1 });

		// rAF stub fires synchronously, so updateMessageAtIndex runs in-band.
		expect(ctx.updateMessageAtIndex).toHaveBeenCalledOnce();
		const [idx, payload] = (ctx.updateMessageAtIndex.mock.calls[0] ?? []) as [
			number,
			{ streamEvents: typeof ctx.streamedEvents }
		];
		expect(idx).toBe(0);
		expect(payload.streamEvents).toHaveLength(1);
		expect(payload.streamEvents[0]).toMatchObject({ type: 'tool_status' });

		// Subsequent in-place mutation of the live accumulator must not be
		// visible in the persisted snapshot — confirms the [...spread] copy.
		ctx.streamedEvents.length = 0;
		expect(payload.streamEvents).toHaveLength(1);
	});

	test('skips stale flush when the assistant message is gone (findMessageIndex < 0)', () => {
		const { ctx } = makeCtx();
		ctx.findMessageIndex = vi.fn(() => -1);
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onToolStatus({ tool: 'web_search', status: 'completed', iteration: 1 });
		expect(ctx.updateMessageAtIndex).not.toHaveBeenCalled();
	});

	test('multiple events in a single rAF window coalesce into one flush', () => {
		const { ctx } = makeCtx();
		// Override the rAF stub for this test to defer until manually run.
		const queue: FrameRequestCallback[] = [];
		vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
			queue.push(cb);
			return queue.length;
		});
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onToolStatus({ tool: 'web_search', status: 'completed', iteration: 1 });
		handlers.onSources({ sources: [{ index: 0, title: 't', url: 'u' }] });
		handlers.onRetraction({ reason: 'r', message: 'm' });
		// Three pushes, but only one rAF scheduled (pendingFlush coalesce).
		expect(queue).toHaveLength(1);
		expect(ctx.updateMessageAtIndex).not.toHaveBeenCalled();
		// Now drain the rAF — single flush carries all three.
		queue[0]!(0);
		expect(ctx.updateMessageAtIndex).toHaveBeenCalledOnce();
		const [, payload] = (ctx.updateMessageAtIndex.mock.calls[0] ?? []) as [
			number,
			{ streamEvents: typeof ctx.streamedEvents }
		];
		expect(payload.streamEvents).toHaveLength(3);
	});
});

describe('createStreamEventHandlers.onToolArtifacts', () => {
	test('image artifact promotes to onAttachments as IMAGE extra and pushes event', () => {
		const { ctx, onAttachments } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);

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
		// Event also accumulated for tool-panel history
		expect(ctx.streamedEvents).toHaveLength(1);
		expect(ctx.streamedEvents[0]).toMatchObject({ type: 'tool_artifacts', offset: 0 });
	});

	test('image artifact without dimensions omits width/height/url keys', () => {
		const { ctx, onAttachments } = makeCtx();
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
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
		const { handlers } = createStreamEventHandlers(ctx as any);
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
		const { handlers } = createStreamEventHandlers(ctx as any);
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

describe('createStreamEventHandlers.onCompaction', () => {
	test('resolves boundary against synthetic compaction-summary placeholder', () => {
		const { ctx } = makeCtx();
		const realBoundaryMsg = { id: 'real-msg-after-summary' } as never;
		ctx.sentMessages = () =>
			[{ id: 'system' }, { id: 'compaction-summary' }, realBoundaryMsg] as never;
		ctx.activeConversation = () => ({ id: 'conv-1' });
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onCompaction({
			summary: 'short summary',
			compacted_up_to_index: 1,
			compacted_message_count: 8
		} as never);
		expect(ctx.setCompaction).toHaveBeenCalledOnce();
		const [arg] = ctx.setCompaction.mock.calls[0];
		expect((arg as { compactedUpToMessageId: string }).compactedUpToMessageId).toBe(
			'real-msg-after-summary'
		);
	});

	test('returns silently when boundary cannot be resolved', () => {
		const { ctx } = makeCtx();
		// sentMessages too short — boundary index out of range
		ctx.sentMessages = () => [{ id: 'system' }] as never;
		ctx.activeConversation = () => ({ id: 'conv-1' });
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const { handlers } = createStreamEventHandlers(ctx as any);
		handlers.onCompaction({
			summary: 's',
			compacted_up_to_index: 5,
			compacted_message_count: 0
		} as never);
		expect(ctx.setCompaction).not.toHaveBeenCalled();
	});
});
