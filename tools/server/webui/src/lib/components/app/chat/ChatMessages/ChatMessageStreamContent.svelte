<script lang="ts">
	import ReasoningBlock from './ReasoningBlock.svelte';
	import ToolTimeline from './ToolTimeline.svelte';
	import RetractionBlock from './RetractionBlock.svelte';
	import SourcesFooter from './SourcesFooter.svelte';
	import { MarkdownContent } from '$lib/components/app';
	import { LEGACY_REASONING_TAGS as REASONING_TAGS, REASONING_BOUNDARY } from '$lib/constants';
	import { config } from '$lib/stores/settings.svelte';
	import type { StreamEvent, DatabaseMessageExtra, ApiToolArtifactPayload } from '$lib/types';
	import type { SourceItem, ToolStep } from '$lib/types/chat';

	interface Props {
		content: string;
		streamEvents: StreamEvent[];
		attachments?: DatabaseMessageExtra[];
		reasoningContent?: string;
		isStreaming?: boolean;
	}

	let { content, streamEvents, attachments, reasoningContent, isStreaming = false }: Props = $props();

	// --- Reasoning: prefer legacy inline markers (old stored messages), fall back to dedicated field ---

	interface ReasoningData {
		initial: string | null;
		initialPending: boolean;
		perStep: Map<number, { content: string; pending: boolean }>;
	}

	let reasoning = $derived.by((): ReasoningData => {
		// Try legacy inline markers (for old stored messages)
		const legacyBlocks: Array<{ content: string; pending: boolean }> = [];
		let cursor = 0;
		while (cursor < content.length) {
			const startIdx = content.indexOf(REASONING_TAGS.START, cursor);
			if (startIdx === -1) break;
			const contentStart = startIdx + REASONING_TAGS.START.length;
			const endIdx = content.indexOf(REASONING_TAGS.END, contentStart);
			if (endIdx === -1) {
				legacyBlocks.push({ content: content.slice(contentStart), pending: true });
				break;
			}
			legacyBlocks.push({ content: content.slice(contentStart, endIdx), pending: false });
			cursor = endIdx + REASONING_TAGS.END.length;
		}

		if (legacyBlocks.length > 0) {
			// Legacy path: block 0 is initial, blocks 1+ map to tool iterations
			const perStep = new Map<number, { content: string; pending: boolean }>();
			for (let i = 1; i < legacyBlocks.length; i++) {
				perStep.set(i - 1, legacyBlocks[i]);
			}
			return {
				initial: legacyBlocks[0].content,
				initialPending: legacyBlocks[0].pending,
				perStep
			};
		}

		// Modern path: dedicated reasoningContent field.
		// Split on boundary markers to recover per-iteration reasoning blocks.
		// Block 0 = initial reasoning (before tools), block N = reasoning after tool iteration N-1.
		if (reasoningContent) {
			const blocks = reasoningContent.split(REASONING_BOUNDARY).filter(b => b.trim());
			const perStep = new Map<number, { content: string; pending: boolean }>();
			for (let i = 1; i < blocks.length; i++) {
				// Last block may still be streaming (pending)
				const isPending = isStreaming && i === blocks.length - 1;
				perStep.set(i - 1, { content: blocks[i], pending: isPending });
			}
			return {
				initial: blocks[0] || null,
				initialPending: false, // Always finalized before first tool_status
				perStep
			};
		}

		return { initial: null, initialPending: false, perStep: new Map() };
	});

	// Content with legacy reasoning stripped
	let strippedContent = $derived(
		content
			.replace(/<<<reasoning_content_start>>>[\s\S]*?<<<reasoning_content_end>>>/g, '')
			.replace(/<<<reasoning_content_start>>>[\s\S]*$/, '')
	);

	// --- Stream event extraction ---

	let sources = $derived.by(() => {
		const sourcesEvent = streamEvents.findLast((e) => e.type === 'sources');
		if (!sourcesEvent) return [];
		const data = sourcesEvent.data as { sources?: SourceItem[]; final?: boolean };
		return data.sources ?? [];
	});

	let sourcesFinal = $derived.by(() => {
		const sourcesEvent = streamEvents.findLast((e) => e.type === 'sources');
		if (!sourcesEvent) return false;
		const data = sourcesEvent.data as { final?: boolean };
		return data.final ?? false;
	});

	let retractionOffset = $derived.by(() => {
		const retractionEvent = streamEvents.find((e) => e.type === 'retraction');
		return retractionEvent?.offset ?? null;
	});

	// --- Retraction: content before offset shown collapsed ---

	let retractedContent = $derived(
		retractionOffset !== null && retractionOffset > 0
			? strippedContent.slice(0, retractionOffset).trim()
			: null
	);

	let retractedReasoning = $derived(
		retractionOffset !== null && reasoning.initial ? reasoning.initial : null
	);

	let effectiveContent = $derived(
		retractionOffset !== null ? strippedContent.slice(retractionOffset) : strippedContent
	);

	// Initial reasoning shown independently only when no retraction
	let initialReasoning = $derived(
		retractionOffset === null ? reasoning.initial : null
	);

	let initialReasoningPending = $derived(
		retractionOffset === null ? reasoning.initialPending : false
	);

	// --- Tool steps: group tool_status events by iteration, attach per-step reasoning ---

	let toolSteps = $derived.by((): ToolStep[] => {
		const groups = new Map<number, StreamEvent[]>();
		for (const event of streamEvents) {
			if (event.type !== 'tool_status') continue;
			const iter = (event.data as { iteration?: number }).iteration ?? 1;
			const existing = groups.get(iter) ?? [];
			existing.push(event);
			groups.set(iter, existing);
		}

		// Build artifact map keyed by call_id (one entry per tool call).
		// Each tool_artifacts SSE event carries a single artifact; multiple
		// events for the same call_id accumulate into the chip's array.
		const artifactMap = new Map<string, ApiToolArtifactPayload[]>();
		for (const event of streamEvents) {
			if (event.type !== 'tool_artifacts') continue;
			const data = event.data as { call_id?: string; artifact?: ApiToolArtifactPayload };
			if (!data.call_id || !data.artifact) continue;
			const existing = artifactMap.get(data.call_id) ?? [];
			artifactMap.set(data.call_id, [...existing, data.artifact]);
		}

		const iterations = [...groups.keys()].sort((a, b) => a - b);
		return iterations.map((iter, i) => {
			const events = groups.get(iter)!;
			const stepReasoning = reasoning.perStep.get(i);
			// Resolve to latest status per tool/call_id
			const latest = new Map<string, { tool: string; status: string; query?: string }>();
			for (const e of events) {
				const data = e.data as { tool?: string; status?: string; query?: string; call_id?: string };
				if (data.tool && data.status) {
					const key = data.call_id || data.tool;
					const existing = latest.get(key);
					latest.set(key, {
						tool: data.tool,
						status: data.status,
						query: data.query || existing?.query
					});
				}
			}
			return {
				reasoning: stepReasoning?.content,
				reasoningPending: stepReasoning?.pending,
				tools: [...latest.entries()].map(([key, info]) => ({
					tool: info.tool,
					status: info.status,
					query: info.query,
					call_id: key,
					artifacts: artifactMap.get(key)
				}))
			};
		});
	});

	let hasToolCalls = $derived(
		toolSteps.reduce((sum, step) => sum + step.tools.length, 0) > 0
	);

	let hasContentSegments = $derived(effectiveContent.trim().length > 0);

	// --- Reasoning toggle state (for initial reasoning only -- step reasoning managed by ToolTimeline) ---

	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	let initialReasoningManualToggle = $state<boolean | null>(null);
	let isInitialReasoningOpen = $derived(
		initialReasoningManualToggle !== null
			? initialReasoningManualToggle
			: (isStreaming && showThoughtInProgress) || initialReasoningPending
	);
</script>

<div class="stream-content">
	<RetractionBlock
		content={retractedContent}
		reasoning={retractedReasoning}
		{attachments}
	/>

	{#if initialReasoning}
		<ReasoningBlock
			content={initialReasoning}
			pending={initialReasoningPending}
			open={isInitialReasoningOpen}
			{isStreaming}
			onToggle={() => {
				initialReasoningManualToggle = !isInitialReasoningOpen;
			}}
		/>
	{/if}

	{#if hasToolCalls}
		<ToolTimeline steps={toolSteps} {isStreaming} />
	{/if}

	{#if hasContentSegments}
		<MarkdownContent
			content={effectiveContent}
			{attachments}
			{sources}
		/>
	{/if}

	{#if isStreaming && !hasContentSegments}
		<div class="streaming-placeholder"></div>
	{/if}

	<SourcesFooter {sources} final={sourcesFinal} />
</div>

<style>
	.stream-content {
		width: 100%;
	}

	.streaming-placeholder {
		height: 1.5rem;
	}
</style>
