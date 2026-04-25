import { describe, test, expect } from 'vitest';
import { AttachmentType } from '$lib/enums';
import type { DatabaseMessageExtra } from '$lib/types';
import { resolveInlineImageSrcs } from '$lib/utils/resolve-inline-image-srcs';

const img = (name: string, base64Url: string): DatabaseMessageExtra => ({
    type: AttachmentType.IMAGE,
    name,
    base64Url,
});

describe('resolveInlineImageSrcs', () => {
    test('empty content returns empty set', () => {
        expect(resolveInlineImageSrcs('', [])).toEqual(new Set());
    });

    test('name-based markdown ref resolves to attachment base64Url', () => {
        const attachments = [img('plot.png', 'data:image/png;base64,AAA')];
        const result = resolveInlineImageSrcs(
            'Here is the plot: ![nice](plot.png) — enjoy.',
            attachments,
        );
        expect(result).toEqual(new Set(['data:image/png;base64,AAA']));
    });

    test('literal URL markdown ref is included as-is', () => {
        const result = resolveInlineImageSrcs(
            '![x](/images/foo.png)',
            [],
        );
        expect(result).toEqual(new Set(['/images/foo.png']));
    });

    test('HTML <img src> is parsed alongside markdown', () => {
        const result = resolveInlineImageSrcs(
            '<img src="/a.png" alt="a"> and ![b](/b.png)',
            [],
        );
        expect(result).toEqual(new Set(['/a.png', '/b.png']));
    });

    test('multiple refs to same name produce one set entry', () => {
        const attachments = [img('p.png', 'data:X')];
        const result = resolveInlineImageSrcs(
            '![first](p.png) and again ![second](p.png)',
            attachments,
        );
        expect(result.size).toBe(1);
        expect(result.has('data:X')).toBe(true);
    });

    test('unmatched name falls through as literal', () => {
        const result = resolveInlineImageSrcs(
            '![x](unknown.png)',
            [img('different.png', 'data:Y')],
        );
        expect(result).toEqual(new Set(['unknown.png']));
    });

    test('markdown image with title does not confuse parser', () => {
        const attachments = [img('p.png', 'data:Z')];
        const result = resolveInlineImageSrcs(
            '![alt](p.png "a title in quotes")',
            attachments,
        );
        expect(result).toEqual(new Set(['data:Z']));
    });

    test('non-IMAGE attachments do not populate the name map', () => {
        const attachments: DatabaseMessageExtra[] = [
            { type: AttachmentType.TEXT, name: 'p.png', content: 'hi' } as any,
        ];
        const result = resolveInlineImageSrcs('![x](p.png)', attachments);
        // Since no IMAGE attachment matched, target 'p.png' falls through literal
        expect(result).toEqual(new Set(['p.png']));
    });

    test('case-insensitive HTML tag match (<IMG SRC="...">)', () => {
        const result = resolveInlineImageSrcs(
            '<IMG SRC="/cap.png">',
            [],
        );
        expect(result).toEqual(new Set(['/cap.png']));
    });
});
