<script lang="ts">
	import { ChatMessage, ChatMessageUserPending } from '$lib/components/app';
	import CompactionBanner from '$lib/components/app/chat/CompactionBanner.svelte';
	import { setChatActionsContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { chatStore } from '$lib/stores/chat.svelte';
	import {
		chatPendingMessageContent,
		chatPendingMessageExtras,
		chatClearPendingMessage,
		chatInjectPendingMessage
	} from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import {
		agenticPendingSteeringMessageContent,
		agenticPendingSteeringMessageExtras,
		agenticClearSteeringMessage,
		agenticInjectSteeringMessage
	} from '$lib/stores/agentic.svelte';
	import {
		copyToClipboard,
		formatMessageForClipboard,
		getMessageSiblings,
		hasAgenticContent
	} from '$lib/utils';

	interface Props {
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { messages = [], onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	let summaryExpanded = $state(false);
	let historyExpanded = $state(false);
	const currentConfig = $derived(config());

	// Reset banner expansion state when conversation changes — otherwise a
	// previous chat's history toggle bleeds into the next one.
	$effect(() => {
		activeConversation();
		summaryExpanded = false;
		historyExpanded = false;
	});

	// Compaction state from active conversation, plus the boundary index in
	// `messages` so the renderer can dim everything from index 0 through the
	// boundary message inclusive.
	let compaction = $derived(activeConversation()?.compaction);
	let compactionBoundaryIndex = $derived.by(() => {
		if (!compaction || !messages.length) return -1;
		return messages.findIndex((m) => m.id === compaction!.compactedUpToMessageId);
	});

	setChatActionsContext({
		copy: async (message: DatabaseMessage) => {
			const asPlainText = Boolean(currentConfig.copyTextAttachmentsAsPlainText);
			const clipboardContent = formatMessageForClipboard(
				message.content,
				message.extra,
				asPlainText
			);
			await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		},

		delete: async (message: DatabaseMessage) => {
			await chatStore.deleteMessage(message.id);
			refreshAllMessages();
		},

		navigateToSibling: async (siblingId: string) => {
			await conversationsStore.navigateToSibling(siblingId);
		},

		editWithBranching: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editMessageWithBranching(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		editWithReplacement: async (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => {
			onUserAction?.();
			await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);
			refreshAllMessages();
		},

		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			onUserAction?.();
			await chatStore.regenerateMessageWithBranching(message.id, modelOverride);
			refreshAllMessages();
		},

		continueAssistantMessage: async (message: DatabaseMessage) => {
			onUserAction?.();
			await chatStore.continueAssistantMessage(message.id);
			refreshAllMessages();
		},

		forkConversation: async (
			message: DatabaseMessage,
			options: { name: string; includeAttachments: boolean }
		) => {
			await conversationsStore.forkConversation(message.id, options);
		}
	});

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();

		if (conversation) {
			refreshAllMessages();
		}
	});

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== MessageRole.SYSTEM);

		// Build display entries, grouping agentic sessions into single entries.
		// An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
		const result: Array<{
			message: DatabaseMessage;
			toolMessages: DatabaseMessage[];
			isLastAssistantMessage: boolean;
			isInCompactedRegion: boolean;
			isCompactionBoundaryNext: boolean;
			siblingInfo: ChatMessageSiblingInfo;
		}> = [];

		for (let i = 0; i < filteredMessages.length; i++) {
			const msg = filteredMessages[i];

			// Skip tool messages - they're grouped with preceding assistant
			if (msg.role === MessageRole.TOOL) continue;

			const toolMessages: DatabaseMessage[] = [];
			if (msg.role === MessageRole.ASSISTANT && hasAgenticContent(msg)) {
				let j = i + 1;

				while (j < filteredMessages.length) {
					const next = filteredMessages[j];

					if (next.role === MessageRole.TOOL) {
						toolMessages.push(next);

						j++;
					} else if (next.role === MessageRole.ASSISTANT) {
						toolMessages.push(next);

						j++;
					} else {
						break;
					}
				}

				i = j - 1;
			} else if (msg.role === MessageRole.ASSISTANT) {
				let j = i + 1;

				while (j < filteredMessages.length && filteredMessages[j].role === MessageRole.TOOL) {
					toolMessages.push(filteredMessages[j]);
					j++;
				}
			}

			const siblingInfo = getMessageSiblings(allConversationMessages, msg.id);

			// Compaction zone membership: messages whose index in the original
			// `messages` array sits at or before the boundary message are dimmed
			// (and hidden unless historyExpanded). The message immediately AFTER
			// the boundary triggers banner rendering above it.
			const origIndex = messages.indexOf(msg);
			const isSystemMessage = msg.type === MessageRole.SYSTEM;
			const isInCompactedRegion =
				compactionBoundaryIndex >= 0 &&
				!isSystemMessage &&
				origIndex >= 0 &&
				origIndex <= compactionBoundaryIndex;
			const isCompactionBoundaryNext =
				compactionBoundaryIndex >= 0 && origIndex === compactionBoundaryIndex + 1;

			result.push({
				message: msg,
				toolMessages,
				isLastAssistantMessage: false,
				isInCompactedRegion,
				isCompactionBoundaryNext,
				siblingInfo: siblingInfo || {
					message: msg,
					siblingIds: [msg.id],
					currentIndex: 0,
					totalSiblings: 1
				}
			});
		}

		// Mark the last assistant message
		for (let i = result.length - 1; i >= 0; i--) {
			if (result[i].message.role === MessageRole.ASSISTANT) {
				result[i].isLastAssistantMessage = true;
				break;
			}
		}

		return result;
	});
</script>

{#each displayMessages as { message, toolMessages, isLastAssistantMessage, isInCompactedRegion, isCompactionBoundaryNext, siblingInfo } (message.id)}
	{#if isCompactionBoundaryNext && compaction}
		<CompactionBanner
			summary={compaction.summary}
			messageCount={compaction.compactedMessageCount}
			{summaryExpanded}
			{historyExpanded}
			onToggleSummary={() => {
				summaryExpanded = !summaryExpanded;
				if (!summaryExpanded) historyExpanded = false;
			}}
			onToggleHistory={() => (historyExpanded = !historyExpanded)}
		/>
	{/if}

	{#if isInCompactedRegion}
		{#if historyExpanded}
			<div class="border-l-2 border-muted-foreground/20 opacity-50">
				<ChatMessage
					class="mx-auto mt-12 w-full max-w-[48rem]"
					{message}
					{toolMessages}
					{isLastAssistantMessage}
					{siblingInfo}
				/>
			</div>
		{/if}
	{:else}
		<ChatMessage
			class="mx-auto mt-12 w-full max-w-[48rem]"
			{message}
			{toolMessages}
			{isLastAssistantMessage}
			{siblingInfo}
		/>
	{/if}
{/each}

{#if activeConversation() && agenticPendingSteeringMessageContent(activeConversation()!.id)}
	{@const convId = activeConversation()!.id}
	{@const pendingContent = agenticPendingSteeringMessageContent(convId)}

	{#if pendingContent}
		<ChatMessageUserPending
			class="mx-auto mt-12 w-full max-w-[48rem]"
			content={pendingContent}
			extras={agenticPendingSteeringMessageExtras(convId)}
			onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
			onEdit={(newContent, extras) => agenticInjectSteeringMessage(convId, newContent, extras)}
			onDelete={() => agenticClearSteeringMessage(convId)}
		/>
	{/if}
{:else if activeConversation() && chatPendingMessageContent(activeConversation()!.id)}
	{@const convId = activeConversation()!.id}
	{@const pendingContent = chatPendingMessageContent(convId)}

	{#if pendingContent}
		<ChatMessageUserPending
			class="mx-auto mt-12 w-full max-w-[48rem]"
			content={pendingContent}
			extras={chatPendingMessageExtras(convId)}
			onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
			onEdit={(newContent, extras) => chatInjectPendingMessage(convId, newContent, extras)}
			onDelete={() => chatClearPendingMessage(convId)}
		/>
	{/if}
{/if}
