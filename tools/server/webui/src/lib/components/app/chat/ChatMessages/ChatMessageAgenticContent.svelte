<script lang="ts">
	import {
		ChatMessageStatistics,
		CollapsibleContentBlock,
		MarkdownContent,
		SyntaxHighlightedCode
	} from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import { Wrench, Loader2, AlertTriangle, Brain, Search, RotateCcw } from '@lucide/svelte';
	import { AgenticSectionType, AttachmentType, FileTypeText } from '$lib/enums';
	import { formatJsonPretty } from '$lib/utils';
	import {
		deriveAgenticSections,
		parseToolResultWithImages,
		type AgenticSection,
		type ToolResultLine
	} from '$lib/utils';
	import type { DatabaseMessage } from '$lib/types/database';
	import type { ChatMessageAgenticTimings, ChatMessageAgenticTurnStats } from '$lib/types/chat';
	import { ChatMessageStatsView } from '$lib/enums';

	interface Props {
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		isStreaming?: boolean;
		highlightTurns?: boolean;
	}

	let { message, toolMessages = [], isStreaming = false, highlightTurns = false }: Props = $props();

	let expandedStates: Record<number, boolean> = $state({});
	let condensedExpanded = $state(false);

	const showToolCallInProgress = $derived(config().showToolCallInProgress as boolean);
	const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);

	const sections = $derived(deriveAgenticSections(message, toolMessages, [], isStreaming));

	// Parse tool results with images
	const sectionsParsed = $derived(
		sections.map((section) => ({
			...section,
			parsedLines: section.toolResult
				? parseToolResultWithImages(section.toolResult, section.toolResultExtras || message?.extra)
				: ([] as ToolResultLine[])
		}))
	);

	// Regex matching orchestrator's web_search status messages (primary + follow-up pools)
	const SEARCH_STATUS_RE =
		/^(Searching|Looking that up|Checking online|Let me search|Also searching|And looking|Plus checking)/i;

	// Condense intermediate tool-calling steps when 2+ iterations detected
	const condensedInfo = $derived.by(() => {
		if (isStreaming) return null;

		// Find all TEXT sections and identify the last one (the final response)
		const textEntries = sectionsParsed
			.map((s, i) => ({ s, i }))
			.filter(({ s }) => s.type === AgenticSectionType.TEXT);

		if (textEntries.length < 2) return null;

		const lastTextIdx = textEntries[textEntries.length - 1].i;

		// Collect intermediate status messages: short, contain "…", not the final response
		type Step = {
			statusText: string;
			reasoningContent?: string;
			statusIdx: number;
			reasoningIdx?: number;
		};
		const steps: Step[] = [];

		for (const { s, i } of textEntries) {
			if (i === lastTextIdx) continue;
			const trimmed = s.content.trim();
			if (trimmed.length < 300 && trimmed.includes('\u2026')) {
				const step: Step = { statusText: trimmed, statusIdx: i };
				// Check for following reasoning block (model's reaction to tool results)
				if (
					i + 1 < sectionsParsed.length &&
					sectionsParsed[i + 1].type === AgenticSectionType.REASONING
				) {
					step.reasoningIdx = i + 1;
					step.reasoningContent = sectionsParsed[i + 1].content;
				}
				steps.push(step);
			}
		}

		if (steps.length < 2) return null;

		// Build set of indices to skip in normal rendering
		const skipIndices = new Set<number>();
		for (const step of steps) {
			skipIndices.add(step.statusIdx);
			if (step.reasoningIdx !== undefined) skipIndices.add(step.reasoningIdx);
		}

		// Determine title: "N web searches" if all steps match search patterns, else "N tool calls"
		const allSearches = steps.every((step) => {
			// For multi-line status texts (batched tool calls), check every line
			const lines = step.statusText.split('\n').filter((l) => l.trim());
			return lines.every((line) => SEARCH_STATUS_RE.test(line.trim()));
		});

		const title = allSearches
			? `${steps.length} web searches`
			: `${steps.length} tool calls`;

		return {
			steps,
			skipIndices,
			insertAtIndex: Math.min(...skipIndices),
			title,
			allSearches
		};
	});

	// Group flat sections into agentic turns
	// A new turn starts when a non-tool section follows a tool section
	const turnGroups = $derived.by(() => {
		const turns: { sections: (typeof sectionsParsed)[number][]; flatIndices: number[] }[] = [];
		let currentTurn: (typeof sectionsParsed)[number][] = [];
		let currentIndices: number[] = [];
		let prevWasTool = false;

		for (let i = 0; i < sectionsParsed.length; i++) {
			const section = sectionsParsed[i];
			const isTool =
				section.type === AgenticSectionType.TOOL_CALL ||
				section.type === AgenticSectionType.TOOL_CALL_PENDING ||
				section.type === AgenticSectionType.TOOL_CALL_STREAMING;

			if (!isTool && prevWasTool && currentTurn.length > 0) {
				turns.push({ sections: currentTurn, flatIndices: currentIndices });
				currentTurn = [];
				currentIndices = [];
			}

			currentTurn.push(section);
			currentIndices.push(i);
			prevWasTool = isTool;
		}

		if (currentTurn.length > 0) {
			turns.push({ sections: currentTurn, flatIndices: currentIndices });
		}

		return turns;
	});

	function getDefaultExpanded(section: AgenticSection): boolean {
		if (
			section.type === AgenticSectionType.TOOL_CALL_PENDING ||
			section.type === AgenticSectionType.TOOL_CALL_STREAMING
		) {
			return showToolCallInProgress;
		}

		if (section.type === AgenticSectionType.REASONING_PENDING) {
			return showThoughtInProgress;
		}

		return false;
	}

	function isExpanded(index: number, section: AgenticSection): boolean {
		if (expandedStates[index] !== undefined) {
			return expandedStates[index];
		}

		return getDefaultExpanded(section);
	}

	function toggleExpanded(index: number, section: AgenticSection) {
		const currentState = isExpanded(index, section);

		expandedStates[index] = !currentState;
	}

	function buildTurnAgenticTimings(stats: ChatMessageAgenticTurnStats): ChatMessageAgenticTimings {
		return {
			turns: 1,
			toolCallsCount: stats.toolCalls.length,
			toolsMs: stats.toolsMs,
			toolCalls: stats.toolCalls,
			llm: stats.llm
		};
	}
</script>

{#snippet renderSection(section: (typeof sectionsParsed)[number], index: number)}
	{#if section.type === AgenticSectionType.TEXT}
		<div class="agentic-text">
			<MarkdownContent content={section.content.trim()} attachments={message?.extra} />
		</div>
	{:else if section.type === AgenticSectionType.TOOL_CALL_STREAMING}
		{@const streamingIcon = isStreaming ? Loader2 : Loader2}
		{@const streamingIconClass = isStreaming ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={streamingIcon}
			iconClass={streamingIconClass}
			title={section.toolName || 'Tool call'}
			subtitle={isStreaming ? '' : 'incomplete'}
			{isStreaming}
			onToggle={() => toggleExpanded(index, section)}
		>
			<div class="pt-3">
				<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
					<span>Arguments:</span>

					{#if isStreaming}
						<Loader2 class="h-3 w-3 animate-spin" />
					{/if}
				</div>
				{#if section.toolArgs}
					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolArgs)}
						language={FileTypeText.JSON}
						maxHeight="20rem"
						class="text-xs"
					/>
				{:else if isStreaming}
					<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
						Receiving arguments...
					</div>
				{:else}
					<div
						class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
					>
						Response was truncated
					</div>
				{/if}
			</div>
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING}
		{@const isPending = section.type === AgenticSectionType.TOOL_CALL_PENDING}
		{@const toolIcon = isPending ? Loader2 : Wrench}
		{@const toolIconClass = isPending ? 'h-4 w-4 animate-spin' : 'h-4 w-4'}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={toolIcon}
			iconClass={toolIconClass}
			title={section.toolName || ''}
			subtitle={isPending ? 'executing...' : undefined}
			isStreaming={isPending}
			onToggle={() => toggleExpanded(index, section)}
		>
			{#if section.toolArgs && section.toolArgs !== '{}'}
				<div class="pt-3">
					<div class="my-3 text-xs text-muted-foreground">Arguments:</div>

					<SyntaxHighlightedCode
						code={formatJsonPretty(section.toolArgs)}
						language={FileTypeText.JSON}
						maxHeight="20rem"
						class="text-xs"
					/>
				</div>
			{/if}

			<div class="pt-3">
				<div class="my-3 flex items-center gap-2 text-xs text-muted-foreground">
					<span>Result:</span>

					{#if isPending}
						<Loader2 class="h-3 w-3 animate-spin" />
					{/if}
				</div>
				{#if section.toolResult}
					<div class="overflow-auto rounded-lg border border-border bg-muted p-4">
						{#each section.parsedLines as line, i (i)}
							<div class="font-mono text-xs leading-relaxed whitespace-pre-wrap">{line.text}</div>
							{#if line.image}
								<img
									src={line.image.base64Url}
									alt={line.image.name}
									class="mt-2 mb-2 h-auto max-w-full rounded-lg"
									loading="lazy"
								/>
							{/if}
						{/each}
					</div>
				{:else if isPending}
					<div class="rounded bg-muted/30 p-2 text-xs text-muted-foreground italic">
						Waiting for result...
					</div>
				{/if}
			</div>
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.REASONING}
		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={Brain}
			title="Reasoning"
			onToggle={() => toggleExpanded(index, section)}
		>
			<div class="pt-3">
				<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
					{section.content}
				</div>
			</div>
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.REASONING_PENDING}
		{@const reasoningTitle = isStreaming ? 'Reasoning...' : 'Reasoning'}
		{@const reasoningSubtitle = isStreaming ? '' : 'incomplete'}

		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2"
			icon={Brain}
			title={reasoningTitle}
			subtitle={reasoningSubtitle}
			{isStreaming}
			onToggle={() => toggleExpanded(index, section)}
		>
			<div class="pt-3">
				<div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
					{section.content}
				</div>
			</div>
		</CollapsibleContentBlock>
	{:else if section.type === AgenticSectionType.RETRACTED}
		<CollapsibleContentBlock
			open={isExpanded(index, section)}
			class="my-2 retracted-block"
			icon={RotateCcw}
			iconClass="h-4 w-4 text-muted-foreground"
			title="Superseded"
			subtitle="re-generated below with search results"
			onToggle={() => toggleExpanded(index, section)}
		>
			<div class="pt-3 opacity-50">
				<MarkdownContent content={section.content.trim()} attachments={message?.extra} />
			</div>
		</CollapsibleContentBlock>
	{/if}
{/snippet}

<div class="agentic-content">
	{#if highlightTurns && turnGroups.length > 1}
		{#each turnGroups as turn, turnIndex (turnIndex)}
			{@const turnStats = message?.timings?.agentic?.perTurn?.[turnIndex]}
			<div class="agentic-turn my-2 hover:bg-muted/80 dark:hover:bg-muted/30">
				<span class="agentic-turn-label">Turn {turnIndex + 1}</span>
				{#each turn.sections as section, sIdx (turn.flatIndices[sIdx])}
					{@render renderSection(section, turn.flatIndices[sIdx])}
				{/each}
				{#if turnStats}
					<div class="turn-stats">
						<ChatMessageStatistics
							promptTokens={turnStats.llm.prompt_n}
							promptMs={turnStats.llm.prompt_ms}
							predictedTokens={turnStats.llm.predicted_n}
							predictedMs={turnStats.llm.predicted_ms}
							agenticTimings={turnStats.toolCalls.length > 0
								? buildTurnAgenticTimings(turnStats)
								: undefined}
							initialView={ChatMessageStatsView.GENERATION}
							hideSummary
						/>
					</div>
				{/if}
			</div>
		{/each}
	{:else}
		{#each sectionsParsed as section, index (index)}
			{#if condensedInfo && condensedInfo.skipIndices.has(index)}
				{#if index === condensedInfo.insertAtIndex}
					{@const condensedIcon = condensedInfo.allSearches ? Search : Wrench}
					<CollapsibleContentBlock
						open={condensedExpanded}
						class="my-2"
						icon={condensedIcon}
						title={condensedInfo.title}
						onToggle={() => (condensedExpanded = !condensedExpanded)}
					>
						<div class="pt-2">
							{#each condensedInfo.steps as step, stepIdx}
								<div class="border-b border-muted/50 py-2 last:border-b-0">
									<div class="text-xs text-muted-foreground whitespace-pre-line">
										{step.statusText}
									</div>
									{#if step.reasoningContent}
										<div class="mt-1.5">
											<CollapsibleContentBlock
												open={isExpanded(-(stepIdx + 1), {
													type: AgenticSectionType.REASONING
												} as AgenticSection)}
												icon={Brain}
												title="Reasoning"
												onToggle={() =>
													toggleExpanded(-(stepIdx + 1), {
														type: AgenticSectionType.REASONING
													} as AgenticSection)}
											>
												<div class="pt-2">
													<div
														class="text-xs leading-relaxed break-words whitespace-pre-wrap"
													>
														{step.reasoningContent}
													</div>
												</div>
											</CollapsibleContentBlock>
										</div>
									{/if}
								</div>
							{/each}
						</div>
					</CollapsibleContentBlock>
				{/if}
			{:else}
				{@render renderSection(section, index)}
			{/if}
		{/each}
	{/if}
</div>

<style>
	.agentic-content {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		width: 100%;
		max-width: 48rem;
	}

	.agentic-text {
		width: 100%;
	}

	.agentic-turn {
		position: relative;
		border: 1.5px dashed var(--muted-foreground);
		border-radius: 0.75rem;
		padding: 1rem;
		transition: background 0.1s;
	}

	.agentic-turn-label {
		position: absolute;
		top: -1rem;
		left: 0.75rem;
		padding: 0 0.375rem;
		background: var(--background);
		font-size: 0.7rem;
		font-weight: 500;
		color: var(--muted-foreground);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.turn-stats {
		margin-top: 0.75rem;
		padding-top: 0.5rem;
		border-top: 1px solid hsl(var(--muted) / 0.5);
	}

	:global(.retracted-block) {
		opacity: 0.6;
	}
</style>
