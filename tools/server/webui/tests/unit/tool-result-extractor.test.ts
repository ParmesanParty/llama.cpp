import { describe, it, expect } from 'vitest';
import { extractToolResult } from '$lib/services/tool-result-extractor';
import type { MCPRawToolCallResult } from '$lib/types/mcp';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeText(text: string): MCPRawToolCallResult['content'][number] {
	return { type: 'text', text };
}

function makeImage(mimeType: string, data: string): MCPRawToolCallResult['content'][number] {
	return { type: 'image', mimeType, data };
}

function makeResource(uri: string, mimeType?: string, text?: string): MCPRawToolCallResult['content'][number] {
	return { type: 'resource', resource: { uri, mimeType, text } };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('extractToolResult', () => {
	// 1. Text-only content: URL regex extraction, content join, dedup
	it('joins multiple text parts and extracts unique URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [
				makeText('See https://example.com/a for details.'),
				makeText('Also https://example.com/b and https://example.com/a again.'),
			],
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe(
			'See https://example.com/a for details.\nAlso https://example.com/b and https://example.com/a again.'
		);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/a');
		expect(urls).toContain('https://example.com/b');
		// Deduplication: 'a' appears twice but should only be in sources once
		expect(urls.filter((u) => u === 'https://example.com/a')).toHaveLength(1);
	});

	// 2. Trailing punctuation strip — dot and bang
	it('strips trailing punctuation from extracted URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [
				makeText('Visit https://example.com/page. for info.'),
				makeText('Also https://example.com/x?q=1!'),
			],
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/page');
		expect(urls).not.toContain('https://example.com/page.');
		expect(urls).toContain('https://example.com/x?q=1');
		expect(urls).not.toContain('https://example.com/x?q=1!');
	});

	// 3. Wikipedia-style URL with balanced parens — closing paren must be preserved
	it('preserves trailing ) when parens are balanced in the URL', () => {
		const url = 'https://en.wikipedia.org/wiki/Rust_(programming_language)';
		const raw: MCPRawToolCallResult = {
			content: [makeText(`See ${url} for more.`)],
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain(url);
		expect(urls).not.toContain('https://en.wikipedia.org/wiki/Rust_(programming_language');
	});

	// 4. _meta.sources short-circuits regex extraction
	it('uses _meta.sources and skips regex extraction when present', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeText('Text with https://should-not-appear.com in it.')],
			_meta: {
				sources: [
					{ url: 'https://meta-source.com', title: 'Meta Source', published: '2024-01-01' },
				],
			},
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://meta-source.com');
		expect(urls).not.toContain('https://should-not-appear.com');
		expect(result.sources[0].title).toBe('Meta Source');
		expect(result.sources[0].published).toBe('2024-01-01');
	});

	// 5. _meta.sources filters invalid entries
	it('filters _meta.sources entries with missing, empty, or non-string url', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [
					// valid
					{ url: 'https://good.com', title: 'Good' },
					// no url field
					{ url: undefined as unknown as string, title: 'No URL' },
					// empty string url
					{ url: '', title: 'Empty URL' },
					// numeric url
					{ url: 12345 as unknown as string, title: 'Numeric URL' },
				],
			},
		};
		const result = extractToolResult(raw);
		expect(result.sources).toHaveLength(1);
		expect(result.sources[0].url).toBe('https://good.com');
	});

	// 6. _meta.sources defaults: title falls back to url; published falls back to null
	it('falls back title to url and published to null when absent in _meta.sources', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [
					{ url: 'https://no-title.com' },
				],
			},
		};
		const result = extractToolResult(raw);
		expect(result.sources).toHaveLength(1);
		expect(result.sources[0].title).toBe('https://no-title.com');
		expect(result.sources[0].published).toBeNull();
	});

	// 7. Image content part — produces [image] in content and a data: artifact
	it('handles image content parts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeImage('image/png', 'abc123')],
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[image]');
		expect(result.artifacts).toHaveLength(1);
		expect(result.artifacts[0].kind).toBe('image');
		expect(result.artifacts[0].mime).toBe('image/png');
		expect(result.artifacts[0].data_url).toBe('data:image/png;base64,abc123');
	});

	// 8. Resource content part with non-http uri — adds artifact, content placeholder
	it('handles resource parts with non-http uri as artifacts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeResource('file:///tmp/report.txt', 'text/plain', 'hello')],
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[resource: file:///tmp/report.txt]');
		expect(result.artifacts).toHaveLength(1);
		expect(result.artifacts[0].kind).toBe('resource');
		expect(result.artifacts[0].uri).toBe('file:///tmp/report.txt');
		expect(result.artifacts[0].text).toBe('hello');
	});

	// 9. Resource content part with http uri — added to sources, NOT to artifacts
	it('handles resource parts with http uri as sources, not artifacts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeResource('https://example.com/doc.pdf')],
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[resource: https://example.com/doc.pdf]');
		expect(result.artifacts).toHaveLength(0);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/doc.pdf');
	});

	// 10. Empty content array — no crash, empty result
	it('returns empty content, sources, and artifacts for empty content array', () => {
		const raw: MCPRawToolCallResult = { content: [] };
		const result = extractToolResult(raw);
		expect(result.content).toBe('');
		expect(result.sources).toHaveLength(0);
		expect(result.artifacts).toHaveLength(0);
	});

	// Bonus: unbalanced trailing ) is stripped
	it('strips unbalanced trailing ) from extracted URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeText('See (https://example.com/page) for info.')],
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		// The URL matched by URL_RE is "https://example.com/page)" — trailing ) is unbalanced
		expect(urls).toContain('https://example.com/page');
		expect(urls).not.toContain('https://example.com/page)');
	});

	// Bonus: _meta.sources coerces snippet to undefined when not a string
	it('coerces non-string snippet to undefined in _meta.sources', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [
					{ url: 'https://example.com', title: 'Test', snippet: 42 as unknown as string },
				],
			},
		};
		const result = extractToolResult(raw);
		expect(result.sources[0].snippet).toBeUndefined();
	});
});
