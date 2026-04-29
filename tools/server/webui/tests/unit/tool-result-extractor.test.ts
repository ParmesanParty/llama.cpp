import { describe, expect, it } from 'vitest';

import { extractToolResult } from '$lib/services/tool-result-extractor';
import type { MCPRawToolCallResult } from '$lib/types/mcp';

function makeText(text: string): MCPRawToolCallResult['content'][number] {
	return { type: 'text', text };
}

function makeImage(mimeType: string, data: string): MCPRawToolCallResult['content'][number] {
	return { type: 'image', mimeType, data };
}

function makeResource(
	uri: string,
	mimeType?: string,
	text?: string
): MCPRawToolCallResult['content'][number] {
	return { type: 'resource', resource: { uri, mimeType, text } };
}

describe('extractToolResult', () => {
	it('joins multiple text parts and extracts unique URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [
				makeText('See https://example.com/a for details.'),
				makeText('Also https://example.com/b and https://example.com/a again.')
			]
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe(
			'See https://example.com/a for details.\nAlso https://example.com/b and https://example.com/a again.'
		);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/a');
		expect(urls).toContain('https://example.com/b');
		expect(urls.filter((u) => u === 'https://example.com/a')).toHaveLength(1);
	});

	it('strips trailing punctuation from extracted URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [
				makeText('Visit https://example.com/page. for info.'),
				makeText('Also https://example.com/x?q=1!')
			]
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/page');
		expect(urls).not.toContain('https://example.com/page.');
		expect(urls).toContain('https://example.com/x?q=1');
		expect(urls).not.toContain('https://example.com/x?q=1!');
	});

	it('preserves trailing ) when parens are balanced in the URL', () => {
		const url = 'https://en.wikipedia.org/wiki/Rust_(programming_language)';
		const raw: MCPRawToolCallResult = {
			content: [makeText(`See ${url} for more.`)]
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain(url);
		expect(urls).not.toContain('https://en.wikipedia.org/wiki/Rust_(programming_language');
	});

	it('uses _meta.sources and skips regex extraction when present', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeText('Text with https://should-not-appear.com in it.')],
			_meta: {
				sources: [
					{ url: 'https://meta-source.com', title: 'Meta Source', published: '2024-01-01' }
				]
			}
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://meta-source.com');
		expect(urls).not.toContain('https://should-not-appear.com');
		expect(result.sources[0].title).toBe('Meta Source');
		expect(result.sources[0].published).toBe('2024-01-01');
	});

	it('filters _meta.sources entries with missing, empty, or non-string url', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [
					{ url: 'https://good.com', title: 'Good' },
					{ url: undefined as unknown as string, title: 'No URL' },
					{ url: '', title: 'Empty URL' },
					{ url: 12345 as unknown as string, title: 'Numeric URL' }
				]
			}
		};
		const result = extractToolResult(raw);
		expect(result.sources).toHaveLength(1);
		expect(result.sources[0].url).toBe('https://good.com');
	});

	it('falls back title to url and published to null when absent in _meta.sources', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [{ url: 'https://no-title.com' }]
			}
		};
		const result = extractToolResult(raw);
		expect(result.sources).toHaveLength(1);
		expect(result.sources[0].title).toBe('https://no-title.com');
		expect(result.sources[0].published).toBeNull();
	});

	it('handles image content parts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeImage('image/png', 'abc123')]
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[image]');
		expect(result.artifacts).toHaveLength(1);
		expect(result.artifacts[0].kind).toBe('image');
		expect(result.artifacts[0].mime).toBe('image/png');
		expect(result.artifacts[0].data_url).toBe('data:image/png;base64,abc123');
	});

	it('handles resource parts with non-http uri as artifacts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeResource('file:///tmp/report.txt', 'text/plain', 'hello')]
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[resource: file:///tmp/report.txt]');
		expect(result.artifacts).toHaveLength(1);
		expect(result.artifacts[0].kind).toBe('resource');
		expect(result.artifacts[0].uri).toBe('file:///tmp/report.txt');
		expect(result.artifacts[0].text).toBe('hello');
	});

	it('handles resource parts with http uri as sources, not artifacts', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeResource('https://example.com/doc.pdf')]
		};
		const result = extractToolResult(raw);
		expect(result.content).toBe('[resource: https://example.com/doc.pdf]');
		expect(result.artifacts).toHaveLength(0);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/doc.pdf');
	});

	it('returns empty content, sources, and artifacts for empty content array', () => {
		const raw: MCPRawToolCallResult = { content: [] };
		const result = extractToolResult(raw);
		expect(result.content).toBe('');
		expect(result.sources).toHaveLength(0);
		expect(result.artifacts).toHaveLength(0);
	});

	it('strips unbalanced trailing ) from extracted URLs', () => {
		const raw: MCPRawToolCallResult = {
			content: [makeText('See (https://example.com/page) for info.')]
		};
		const result = extractToolResult(raw);
		const urls = result.sources.map((s) => s.url);
		expect(urls).toContain('https://example.com/page');
		expect(urls).not.toContain('https://example.com/page)');
	});

	it('coerces non-string snippet to undefined in _meta.sources', () => {
		const raw: MCPRawToolCallResult = {
			content: [],
			_meta: {
				sources: [{ url: 'https://example.com', title: 'Test', snippet: 42 as unknown as string }]
			}
		};
		const result = extractToolResult(raw);
		expect(result.sources[0].snippet).toBeUndefined();
	});
});
