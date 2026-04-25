import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types';

/**
 * Given a message's raw markdown content and its attachments, compute the
 * SET of final resolved `<img src>` values that will appear inline after
 * markdown rendering.
 *
 * Two resolution paths produce inline images:
 *   1. Name-based: `![alt](filename)` where `filename` matches an
 *      attachment's `name` → resolved to `attachment.base64Url`
 *   2. Literal: `![alt](url)` where `url` is an http(s) or data: URL →
 *      used as-is
 *
 * Also catches `<img src="...">` for HTML-style embedding.
 *
 * The output set is suitable for direct comparison against
 * `attachment.base64Url` (for name-resolved refs) or `attachment.url`
 * (for literal-URL refs) when deduping the message-level artifact row
 * against inline markdown images.
 */
export function resolveInlineImageSrcs(
    content: string,
    attachments: DatabaseMessageExtra[],
): Set<string> {
    const byName = new Map<string, DatabaseMessageExtraImageFile>();
    for (const a of attachments) {
        if (a.type === AttachmentType.IMAGE) byName.set(a.name, a);
    }

    const srcs = new Set<string>();

    // Markdown image: ![alt](target "optional title")
    const mdRe = /!\[[^\]]*\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)/g;
    // HTML <img ... src="..." ...>
    const htmlRe = /<img\b[^>]*\bsrc\s*=\s*["']([^"']+)["']/gi;

    for (const re of [mdRe, htmlRe]) {
        let m: RegExpExecArray | null;
        while ((m = re.exec(content)) !== null) {
            const target = m[1];
            const named = byName.get(target);
            srcs.add(named ? named.base64Url : target);
        }
    }

    return srcs;
}
