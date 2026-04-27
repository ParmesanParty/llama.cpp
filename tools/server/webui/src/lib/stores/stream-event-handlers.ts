/**
 * Factory for stream event handler callbacks.
 *
 * Extracted from chat.svelte.ts to keep that heavily-upstream-modified file's
 * parmesan diff surface small. The factory closes over a mutable
 * StreamEventContext, accumulates typed SSE events into ctx.streamedEvents,
 * and flushes coalesced UI updates via requestAnimationFrame.
 *
 * Per-event semantics:
 *   onToolStatus     pushEvent + finalizeReasoning when status === 'executing'
 *                    (closes the open reasoning block before tool execution
 *                    so the renderer can split per-iteration reasoning).
 *   onRetraction     pushEvent at current content offset.
 *   onSources        pushEvent at offset 0 (footer is sticky-bottom).
 *   onToolHealth     console.warn only — does NOT accumulate (tool health is
 *                    a sidecar concern handled by toolHealthStore via
 *                    /api/events, not per-message).
 *   onCompaction     resolves the boundary message id against sentMessages
 *                    (which may include the synthetic "compaction-summary"
 *                    placeholder) and stamps it on the conversation.
 *   onToolArtifacts  pushEvent (kept for tool-panel history / debug rendering)
 *                    AND promotes image artifacts into message.extra via the
 *                    injected onAttachments handler so the artifact-row footer
 *                    and inline-markdown image refs can resolve them.
 */

import { AttachmentType } from '$lib/enums';
import type {
	ApiToolStatusEvent,
	ApiToolArgStreamEvent,
	ApiRetractionEvent,
	ApiSourcesEvent,
	ApiToolHealthEvent,
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
	/** Mutable array — handlers push into this; clear via .length = 0 on turn boundaries. */
	streamedEvents: StreamEvent[];
	/** Visible content length excluding reasoning markers — used as event.offset. */
	contentOffset: () => number;
	/** Close the open reasoning block before tool execution begins. */
	finalizeReasoning: () => void;
	/** Flush pending streamedContent into the active message UI. */
	updateStreamingContent: () => void;
	findMessageIndex: (id: string) => number;
	updateMessageAtIndex: (idx: number, data: Partial<DatabaseMessage>) => void;
	/** Read at flush time (mutates as agentic flow advances through turns). */
	getCurrentMessageId: () => string;
	sentMessages: () => DatabaseMessage[] | null;
	activeMessages: () => DatabaseMessage[];
	activeConversation: () => { id: string } | null;
	setCompaction: (compaction: ConversationCompaction) => void;
	/** Append extras onto a message and persist (shared with the upstream agentic onAttachments). */
	onAttachments?: (messageId: string, extras: DatabaseMessageExtra[]) => void;
}

/**
 * Create stream event handler callbacks. The returned object is shaped to
 * spread into ChatStreamCallbacks. Caller is responsible for passing the
 * live `currentMessageId` via ctx.getCurrentMessageId — the agentic flow
 * mutates it across turns as new assistant messages are created.
 *
 * Returns a `cancelPendingFlush` helper alongside the handlers so the chat
 * store can cancel a queued rAF before turn-boundary state mutation
 * (createAssistantMessage clears `streamedEvents.length = 0` and advances
 * `currentMessageId`). Without that cancellation a pending flush fires
 * AFTER the reset and writes an empty event array to the new turn's
 * message, stomping any events accumulated for that turn before the rAF
 * paints.
 */
export function createStreamEventHandlers(ctx: StreamEventContext) {
	let pendingFlush = false;
	let pendingHandle: number | null = null;

	function flushEvents() {
		pendingHandle = null;
		if (!pendingFlush) return;
		pendingFlush = false;
		const idx = ctx.findMessageIndex(ctx.getCurrentMessageId());
		if (idx < 0) return; // Message gone — skip stale flush
		ctx.updateMessageAtIndex(idx, { streamEvents: [...ctx.streamedEvents] });
	}

	function pushEvent(event: StreamEvent) {
		ctx.streamedEvents.push(event);
		if (!pendingFlush) {
			pendingFlush = true;
			pendingHandle = requestAnimationFrame(flushEvents);
		}
	}

	function cancelPendingFlush(): void {
		if (pendingHandle !== null) {
			cancelAnimationFrame(pendingHandle);
			pendingHandle = null;
		}
		pendingFlush = false;
	}

	const handlers = {
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
			console.warn(`[tool-health] ${event.tool}: ${event.state} — ${event.reason}`);
		},
		onCompaction: (metadata: ApiCompactionMetadata) => {
			const msgs = ctx.sentMessages() ?? ctx.activeMessages();
			const boundaryMsg = msgs[metadata.compacted_up_to_index];
			if (!boundaryMsg || !ctx.activeConversation()) return;
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
		},
		onToolArtifacts: (event: ApiToolArtifactsEvent) => {
			pushEvent({
				type: 'tool_artifacts',
				offset: 0,
				data: event as unknown as Record<string, unknown>
			});

			if (event.artifact.kind !== 'image') return;
			const a = event.artifact;
			const extra: DatabaseMessageExtraImageFile = {
				type: AttachmentType.IMAGE,
				name: a.name,
				base64Url: `data:${a.mime};base64,${a.data_b64}`,
				...(a.alt !== undefined ? { alt: a.alt } : {}),
				...(a.width !== undefined ? { width: a.width } : {}),
				...(a.height !== undefined ? { height: a.height } : {}),
				...(a.url !== undefined ? { url: a.url } : {})
			};
			ctx.onAttachments?.(ctx.getCurrentMessageId(), [extra]);
		},
		onToolArgStream: (event: ApiToolArgStreamEvent) => {
			// Accumulate per-call_id started/delta/completed events.  The
			// chip renderer reduces them via buildArgStreamMap into a
			// per-call writing-phase state for the in-flight chip body.
			pushEvent({
				type: 'tool_arg_stream',
				offset: 0,
				data: event as unknown as Record<string, unknown>
			});
		}
	};

	return { handlers, cancelPendingFlush };
}
