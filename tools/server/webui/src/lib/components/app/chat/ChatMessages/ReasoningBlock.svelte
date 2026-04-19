<script lang="ts">
  import { Brain } from '@lucide/svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';

  interface Props {
    content: string;
    pending?: boolean;
    open?: boolean;
    onToggle?: () => void;
    isStreaming?: boolean;
  }

  let {
    content,
    pending = false,
    open = $bindable(false),
    onToggle,
    isStreaming = false
  }: Props = $props();

  let title = $derived(pending && isStreaming ? 'Reasoning...' : 'Reasoning');
  let subtitle = $derived(pending && !isStreaming ? 'incomplete' : '');
</script>

{#if content}
  <CollapsibleContentBlock
    bind:open
    class="my-2"
    icon={Brain}
    {title}
    {subtitle}
    isStreaming={pending && isStreaming}
    {onToggle}
  >
    <div class="pt-3">
      <div class="text-xs leading-relaxed break-words whitespace-pre-wrap">
        {content}
      </div>
    </div>
  </CollapsibleContentBlock>
{/if}
