/**
 * Rehype plugin to resolve numeric [N] citations to links using a source map.
 *
 * Walks the HAST tree and replaces [N] text patterns (outside code/pre) with
 * <a> link elements pointing to the corresponding source URL.
 */

import type { Plugin } from 'unified';
import type { Root, Element, Text, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import type { SourceItem } from '$lib/types/chat';

/** @deprecated Use SourceItem from '$lib/types/chat' directly. */
export type CitationSource = SourceItem;

const CITATION_RE = /\[(\d+)\]/g;

/**
 * Rehype plugin that resolves [N] citation patterns to anchor links.
 * Detects 0-indexed vs 1-indexed by presence of [0] in the full tree text.
 * Out-of-range citations are stripped silently.
 *
 * Usage: pipeline.use(rehypeResolveCitations, sources)
 */
export const rehypeResolveCitations: Plugin<[SourceItem[]], Root> = (sources) => {
	return (tree: Root) => {
		if (!sources.length) return;

		// Build source lookup by index
		const sourceMap = new Map<number, SourceItem>();
		for (const s of sources) {
			sourceMap.set(s.index, s);
		}

		// Detect 0-indexed vs 1-indexed by checking if [0] appears in text
		let hasZero = false;
		visit(tree, 'text', (node: Text) => {
			if (node.value.includes('[0]')) hasZero = true;
		});

		visit(tree, 'element', (node: Element) => {
			// Skip code, pre, and anchor elements (anchors may contain [N] text from prior resolution)
			if (node.tagName === 'code' || node.tagName === 'pre' || node.tagName === 'a') return 'skip';

			const newChildren: ElementContent[] = [];
			let modified = false;

			for (const child of node.children) {
				if (child.type !== 'text') {
					newChildren.push(child);
					continue;
				}

				const text = child.value;
				let lastIndex = 0;
				let match: RegExpExecArray | null;
				const parts: ElementContent[] = [];

				CITATION_RE.lastIndex = 0;
				while ((match = CITATION_RE.exec(text)) !== null) {
					const citNum = parseInt(match[1], 10);
					// Normalize to 0-indexed for lookup
					const lookupIdx = hasZero ? citNum : citNum - 1;
					const source = sourceMap.get(lookupIdx);

					// Add text before match
					if (match.index > lastIndex) {
						parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
					}

					if (source) {
						// Resolved citation → superscript link
						parts.push({
							type: 'element',
							tagName: 'a',
							properties: {
								href: source.url,
								title: source.title,
								className: ['citation-link'],
								target: '_blank',
								rel: 'noopener noreferrer'
							},
							children: [{ type: 'text', value: `[${citNum}]` }]
						});
					}
					// Out-of-range citations are stripped (nothing pushed)

					lastIndex = match.index + match[0].length;
				}

				if (parts.length > 0) {
					// Add remaining text
					if (lastIndex < text.length) {
						parts.push({ type: 'text', value: text.slice(lastIndex) });
					}
					newChildren.push(...parts);
					modified = true;
				} else {
					newChildren.push(child);
				}
			}

			if (modified) {
				node.children = newChildren;
			}
		});
	};
};
