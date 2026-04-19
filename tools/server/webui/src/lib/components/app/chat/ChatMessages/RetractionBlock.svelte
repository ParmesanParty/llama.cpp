<script lang="ts">
  import { TriangleAlert } from '@lucide/svelte';
  import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
  import type { DatabaseMessageExtra } from '$lib/types';

  interface Props {
    content?: string | null;
    reasoning?: string | null;
    attachments?: DatabaseMessageExtra[];
  }

  let { content = null, reasoning = null, attachments }: Props = $props();

  let hasContent = $derived(!!content || !!reasoning);
  let open = $state(false);
</script>

{#if hasContent}
  <div class="retracted-block">
    <CollapsibleContentBlock
      {open}
      class="my-2"
      icon={TriangleAlert}
      iconClass="h-3.5 w-3.5"
      title="Unverified response retracted{open ? '' : ' — expand to view'}"
      onToggle={() => (open = !open)}
    >
      <div class="retracted-body">
        {#if reasoning}
          <div class="retracted-reasoning">
            {reasoning}
          </div>
        {/if}
        {#if content}
          <div class="retracted-content">
            <MarkdownContent content={content} {attachments} />
          </div>
        {/if}
      </div>
    </CollapsibleContentBlock>
  </div>
{/if}

<style>
  .retracted-block {
    opacity: 0.5;
    transition: opacity 0.2s;
  }

  .retracted-block:hover {
    opacity: 0.7;
  }

  .retracted-body {
    padding-top: 0.5rem;
    opacity: 0.6;
  }

  .retracted-reasoning {
    font-size: 0.75rem;
    line-height: 1.5;
    color: var(--muted-foreground);
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px dashed var(--border);
  }

  .retracted-content {
    opacity: 0.8;
  }
</style>
