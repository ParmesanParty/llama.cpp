import type { MCPRawToolCallResult } from '$lib/types/mcp';

export interface ExtractedToolResult {
	content: string;
	sources: Array<{
		url: string;
		title: string;
		published: string | null;
		snippet?: string;
	}>;
	artifacts: Array<{
		kind: 'image' | 'resource';
		mime?: string;
		data_url?: string;
		uri?: string;
		text?: string;
	}>;
}

const URL_RE = /https?:\/\/[^\s<>"]+/g;

/**
 * Strip trailing punctuation from a URL matched by URL_RE.
 * Non-paren punctuation (.,;:!?) is always stripped. A trailing ')' is only
 * stripped when unbalanced (more ')' than '(' in the matched string). After
 * each unbalanced ')' is dropped, non-paren punctuation is re-checked so
 * sequences like "…page).") are fully cleaned.
 */
function stripTrailingPunctuation(url: string): string {
	let stripped = url.replace(/[.,;:!?]+$/, '');
	while (stripped.endsWith(')')) {
		const opens = (stripped.match(/\(/g) ?? []).length;
		const closes = (stripped.match(/\)/g) ?? []).length;
		if (closes > opens) {
			stripped = stripped.slice(0, -1);
			stripped = stripped.replace(/[.,;:!?]+$/, '');
		} else {
			break;
		}
	}
	return stripped;
}

export function extractToolResult(raw: MCPRawToolCallResult): ExtractedToolResult {
	const textParts: string[] = [];
	const artifacts: ExtractedToolResult['artifacts'] = [];

	for (const part of raw.content ?? []) {
		if (part.type === 'text' && part.text != null) {
			textParts.push(part.text);
		} else if (part.type === 'image') {
			textParts.push('[image]');
			if (part.data && part.mimeType) {
				artifacts.push({
					kind: 'image',
					mime: part.mimeType,
					data_url: `data:${part.mimeType};base64,${part.data}`
				});
			}
		} else if (part.type === 'resource' && part.resource) {
			textParts.push(`[resource: ${part.resource.uri}]`);
			if (!part.resource.uri.startsWith('http')) {
				artifacts.push({
					kind: 'resource',
					uri: part.resource.uri,
					mime: part.resource.mimeType,
					text: part.resource.text
				});
			}
		}
	}

	const content = textParts.join('\n');

	let sources: ExtractedToolResult['sources'];
	if (raw._meta?.sources && Array.isArray(raw._meta.sources)) {
		// _meta.sources comes from an untrusted MCP server — narrow each entry at runtime.
		sources = raw._meta.sources
			.filter(
				(s): s is Record<string, unknown> & { url: string } =>
					typeof s === 'object' &&
					s !== null &&
					typeof (s as { url?: unknown }).url === 'string' &&
					(s as { url: string }).url.length > 0
			)
			.map((s) => ({
				url: s.url,
				title: typeof s.title === 'string' && s.title.length > 0 ? (s.title as string) : s.url,
				published:
					typeof s.published === 'string' && s.published.length > 0
						? (s.published as string)
						: null,
				snippet: typeof s.snippet === 'string' ? (s.snippet as string) : undefined
			}));
	} else {
		// Heuristic URL extraction from text and resource parts.
		const seen = new Set<string>();
		sources = [];
		for (const part of raw.content ?? []) {
			if (part.type === 'text' && part.text) {
				for (const m of part.text.matchAll(URL_RE)) {
					const url = stripTrailingPunctuation(m[0]);
					if (!seen.has(url)) {
						seen.add(url);
						sources.push({ url, title: url, published: null });
					}
				}
			}
			if (part.type === 'resource' && part.resource?.uri.startsWith('http')) {
				if (!seen.has(part.resource.uri)) {
					seen.add(part.resource.uri);
					sources.push({
						url: part.resource.uri,
						title: part.resource.uri,
						published: null
					});
				}
			}
		}
	}

	return { content, sources, artifacts };
}
