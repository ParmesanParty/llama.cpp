/**
 * Factory for stream event handler callbacks.
 *
 * Extracted from chat.svelte.ts to reduce the parmesan diff surface on that
 * heavily-upstream-modified file. Uses requestAnimationFrame batching to
 * avoid O(n²) array copies when many events arrive in a single frame.
 */

import { AttachmentType } from '$lib/enums';
import type {
	ApiToolStatusEvent,
	ApiRetractionEvent,
	ApiSourcesEvent,
	ApiToolArtifactsEvent,
	ApiCompactionMetadata,
	StreamEvent,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile,
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
	onAttachments?: (messageId: string, extras: DatabaseMessageExtra[]) => void;
}

/**
 * Create stream event handler callbacks.
 *
 * Returns an object with onToolStatus, onRetraction, onSources,
 * and onCompaction — ready to spread into ChatStreamCallbacks.
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
		onToolArtifacts: (event: ApiToolArtifactsEvent) => {
			// (1) Keep the raw event in streamEvents for tool-panel history / debug rendering.
			pushEvent({
				type: 'tool_artifacts',
				offset: 0,
				data: event as unknown as Record<string, unknown>
			});

			// (2) Promote image artifacts into message.extra so the message-level
			// artifact row picks them up and inline markdown can resolve to the
			// base64 payload. Non-image kinds short-circuit (future kinds land
			// in separate PRs).
			if (event.artifact.kind !== 'image') return;
			const a = event.artifact;
			const extra: DatabaseMessageExtraImageFile = {
				type: AttachmentType.IMAGE,
				name: a.name,
				base64Url: `data:${a.mime};base64,${a.data_b64}`,
				...(a.width !== undefined ? { width: a.width } : {}),
				...(a.height !== undefined ? { height: a.height } : {}),
				...(a.url !== undefined ? { url: a.url } : {})
			};
			ctx.onAttachments?.(ctx.assistantMessageId, [extra]);
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
