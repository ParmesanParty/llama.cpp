<script lang="ts">
  import { SvelteMap } from 'svelte/reactivity';
  import { Brain, Wrench } from '@lucide/svelte';
  import ToolStatusChip from './ToolStatusChip.svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';
  import { config } from '$lib/stores/settings.svelte';
  import type { ToolStep, ToolChip } from '$lib/types/chat';

  interface Props {
    steps: ToolStep[];
    isStreaming?: boolean;
  }

  let { steps, isStreaming = false }: Props = $props();

  let totalToolCalls = $derived(
    steps.reduce((sum, step) => sum + step.tools.length, 0)
  );

  // --- Collapsible state ---

  let timelineManualToggle = $state<boolean | null>(null);
  let isTimelineOpen = $derived(
    timelineManualToggle !== null ? timelineManualToggle : isStreaming
  );
  function toggleTimeline() {
    timelineManualToggle = !isTimelineOpen;
  }

  // Reasoning toggles per step index
  const showThoughtInProgress = $derived(config().showThoughtInProgress as boolean);
  let reasoningToggles = new SvelteMap<number, boolean>();

  function isReasoningOpen(index: number, pending?: boolean): boolean {
    const manual = reasoningToggles.get(index);
    if (manual !== undefined) return manual;
    return (isStreaming && showThoughtInProgress) || (pending ?? false);
  }

  function toggleReasoning(index: number, pending?: boolean) {
    reasoningToggles.set(index, !isReasoningOpen(index, pending));
  }

  // Chip expansion state: keyed by "stepIdx-{call_id || tool}"
  let chipToggles = new SvelteMap<string, boolean>();

  function isChipExpanded(stepIdx: number, chip: ToolChip): boolean {
    return chipToggles.get(`${stepIdx}-${chip.call_id || chip.tool}`) ?? false;
  }

  function toggleChip(stepIdx: number, chip: ToolChip) {
    const key = `${stepIdx}-${chip.call_id || chip.tool}`;
    chipToggles.set(key, !chipToggles.get(key));
  }

  // Expand/collapse all: mutate SvelteMaps in place
  function expandAll() {
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].reasoning) reasoningToggles.set(i, true);
      for (const chip of steps[i].tools) {
        if (chip.query || chip.argStream) chipToggles.set(`${i}-${chip.call_id || chip.tool}`, true);
      }
    }
  }

  function collapseAll() {
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].reasoning) reasoningToggles.set(i, false);
      for (const chip of steps[i].tools) {
        if (chip.query || chip.argStream) chipToggles.set(`${i}-${chip.call_id || chip.tool}`, false);
      }
    }
  }

  let allExpanded = $derived.by(() => {
    let count = 0;
    let expandedCount = 0;
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].reasoning) {
        count++;
        if (reasoningToggles.get(i)) expandedCount++;
      }
      for (const chip of steps[i].tools) {
        if (chip.query || chip.argStream) {
          count++;
          if (chipToggles.get(`${i}-${chip.call_id || chip.tool}`)) expandedCount++;
        }
      }
    }
    return count > 0 && expandedCount === count;
  });
</script>

{#snippet stepContent(step: ToolStep, i: number, nested: boolean)}
  <div class="tool-step">
    <div class="tool-status-row" class:nested class:first-step={nested && i === 0}>
      {#each step.tools as chip}
        <ToolStatusChip
          tool={chip.tool}
          status={chip.status}
          query={chip.query}
          expanded={isChipExpanded(i, chip)}
          onToggleExpand={() => toggleChip(i, chip)}
          artifacts={chip.artifacts}
          argStream={chip.argStream}
        />
      {/each}
    </div>
    {#if step.reasoning}
      <button class="step-reasoning-toggle" onclick={() => toggleReasoning(i, step.reasoningPending)}>
        <Brain class="step-reasoning-icon" size={12} />
        <span class="step-reasoning-preview">
          {#if isReasoningOpen(i, step.reasoningPending)}
            hide reasoning
          {:else}
            {step.reasoning.slice(0, 80)}{step.reasoning.length > 80 ? '…' : ''}
          {/if}
        </span>
      </button>
      {#if isReasoningOpen(i, step.reasoningPending)}
        <div class="step-reasoning-content">
          {step.reasoning}
        </div>
      {/if}
    {/if}
  </div>
{/snippet}

{#if totalToolCalls > 0}
  {#if isStreaming}
    {#each steps as step, i}
      {@render stepContent(step, i, false)}
    {/each}
  {:else}
    <CollapsibleContentBlock
      open={isTimelineOpen}
      class="my-2"
      icon={Wrench}
      title="{totalToolCalls} Tool Call{totalToolCalls !== 1 ? 's' : ''}"
      onToggle={toggleTimeline}
    >
      <div class="tool-calls-body">
        <button class="expand-all-btn" onclick={() => allExpanded ? collapseAll() : expandAll()}>
          {allExpanded ? 'collapse all' : 'expand all'}
        </button>
        {#each steps as step, i}
          {@render stepContent(step, i, true)}
        {/each}
      </div>
    </CollapsibleContentBlock>
  {/if}
{/if}

<style>
  .tool-step {
    margin: 0.375rem 0;
  }

  .tool-status-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem;
    align-items: flex-start;
  }

  .tool-status-row.nested {
    margin: 0;
  }

  .tool-status-row.first-step {
    padding-right: 5.5rem;
  }

  .step-reasoning-toggle {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.125rem 0;
    margin-top: 0.125rem;
    background: none;
    border: none;
    cursor: pointer;
    color: var(--muted-foreground);
    font-size: 0.75rem;
    line-height: 1.4;
    text-align: left;
    opacity: 0.7;
    transition: opacity 0.15s;
  }

  .step-reasoning-toggle:hover {
    opacity: 1;
  }

  .step-reasoning-toggle :global(.step-reasoning-icon) {
    flex-shrink: 0;
    opacity: 0.6;
  }

  .step-reasoning-preview {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .step-reasoning-content {
    font-size: 0.75rem;
    line-height: 1.5;
    color: var(--muted-foreground);
    padding: 0.25rem 0 0.25rem 1.125rem;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .tool-calls-body {
    position: relative;
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
  }

  .expand-all-btn {
    position: absolute;
    top: 0.25rem;
    right: 0;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 0.6875rem;
    color: var(--muted-foreground);
    opacity: 0.6;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    transition: opacity 0.15s;
    z-index: 1;
  }

  .expand-all-btn:hover {
    opacity: 1;
    background: var(--muted);
  }
</style>
