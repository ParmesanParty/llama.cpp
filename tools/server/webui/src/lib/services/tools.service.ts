import { apiFetch } from '$lib/utils';
import { API_TOOLS } from '$lib/constants';
import { ToolResponseField } from '$lib/enums';
import type { MCPRawToolCallResult, ToolExecutionResult, ServerBuiltinToolInfo } from '$lib/types';

/**
 * Wrap a builtin-tool flat response in a minimal MCP-shape raw result so the
 * ToolExecutionResult contract holds. Builtin sources/artifacts arrive via
 * typed SSE events, not _meta — so the synthesized raw carries content only.
 */
function synthesizeRaw(content: string, isError: boolean): MCPRawToolCallResult {
	return {
		content: [{ type: 'text', text: content }],
		isError
	};
}

export class ToolsService {
	/**
	 * Fetch the list of built-in tools from the server.
	 *
	 * @returns Array of tool definitions in OpenAI-compatible format
	 */
	static async list(): Promise<ServerBuiltinToolInfo[]> {
		return apiFetch<ServerBuiltinToolInfo[]>(API_TOOLS.LIST);
	}

	/**
	 * Execute a built-in tool on the server.
	 */
	static async executeTool(
		toolName: string,
		params: Record<string, unknown>,
		signal?: AbortSignal
	): Promise<ToolExecutionResult> {
		const result = await apiFetch<Record<string, unknown>>(API_TOOLS.EXECUTE, {
			method: 'POST',
			body: JSON.stringify({ tool: toolName, params }),
			signal
		});

		if (ToolResponseField.ERROR in result) {
			const content = String(result[ToolResponseField.ERROR]);
			return { content, isError: true, raw: synthesizeRaw(content, true) };
		}

		if (ToolResponseField.PLAIN_TEXT in result) {
			const content = String(result[ToolResponseField.PLAIN_TEXT]);
			return { content, isError: false, raw: synthesizeRaw(content, false) };
		}

		const content = JSON.stringify(result);
		return { content, isError: false, raw: synthesizeRaw(content, false) };
	}
}
