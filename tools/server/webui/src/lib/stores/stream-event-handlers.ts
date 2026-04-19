/**
 * Factory for stream event handler callbacks.
 *
 * Extracted from chat.svelte.ts to reduce the parmesan diff surface on that
 * heavily-upstream-modified file. Uses requestAnimationFrame batching to
 * avoid O(n²) array copies when many events arrive in a single frame.
 */

import type {
	ApiToolStatusEvent,
	ApiRetractionEvent,
	ApiSourcesEvent,
	ApiToolHealthEvent,
	ApiCompactionMetadata,
	StreamEvent,
	DatabaseMessage,
	ConversationCompaction
} from '$lib/types';

/** Mutable context shared between the stream event handlers and the chat store. */
export interface StreamEventContext {
	streamedEvents: StreamEvent[];
	contentOffset: () => number;
	finalizeReasoning: () => void;
	updateStreamingContent: () => void;
	findMessageIndex: (id: string) => number;
	updateMessageAtIndex: (idx: number, data: Partial<DatabaseMessage>) => void;
	assistantMessageId: string;
	sentMessages: () => DatabaseMessage[] | null;
	activeMessages: () => DatabaseMessage[];
	activeConversation: () => { id: string } | null;
	setCompaction: (compaction: ConversationCompaction) => void;
}

/**
 * Create stream event handler callbacks.
 *
 * Returns an object with onToolStatus, onRetraction, onSources,
 * onToolHealth, and onCompaction — ready to spread into ChatStreamCallbacks.
 */
export function createStreamEventHandlers(ctx: StreamEventContext) {
	let pendingFlush = false;

	function flushEvents() {
		if (!pendingFlush) return;
		pendingFlush = false;
		const idx = ctx.findMessageIndex(ctx.assistantMessageId);
		if (idx < 0) return; // Message gone — skip stale flush
		ctx.updateMessageAtIndex(idx, { streamEvents: [...ctx.streamedEvents] });
	}

	function pushEvent(event: StreamEvent) {
		ctx.streamedEvents.push(event);
		if (!pendingFlush) {
			pendingFlush = true;
			requestAnimationFrame(flushEvents);
		}
	}

	return {
		onToolStatus: (event: ApiToolStatusEvent) => {
			if (event.status === 'executing') {
				ctx.finalizeReasoning();
				ctx.updateStreamingContent();
			}
			pushEvent({
				type: 'tool_status',
				offset: ctx.contentOffset(),
				data: event as unknown as Record<string, unknown>
			});
		},
		onRetraction: (event: ApiRetractionEvent) => {
			pushEvent({
				type: 'retraction',
				offset: ctx.contentOffset(),
				data: event as unknown as Record<string, unknown>
			});
		},
		onSources: (event: ApiSourcesEvent) => {
			pushEvent({
				type: 'sources',
				offset: 0,
				data: event as unknown as Record<string, unknown>
			});
		},
		onToolHealth: (event: ApiToolHealthEvent) => {
			console.warn(
				`[tool-health] ${event.tool}: ${event.state} — ${event.reason}`
			);
		},
		onCompaction: (metadata: ApiCompactionMetadata) => {
			const msgs = ctx.sentMessages() ?? ctx.activeMessages();
			const boundaryMsg = msgs[metadata.compacted_up_to_index];
			if (boundaryMsg && ctx.activeConversation()) {
				const resolvedId =
					boundaryMsg.id === 'compaction-summary'
						? msgs[metadata.compacted_up_to_index + 1]?.id
						: boundaryMsg.id;
				if (!resolvedId) return;
				ctx.setCompaction({
					summary: metadata.summary,
					compactedUpToMessageId: resolvedId,
					compactedMessageCount: metadata.compacted_message_count,
					timestamp: Date.now()
				});
			}
		}
	};
}
