import { describe, it, expect } from 'vitest';
import { buildArgStreamMap } from '$lib/utils/arg-stream-aggregator';
import type { StreamEvent } from '$lib/types';

function ev(data: Record<string, unknown>): StreamEvent {
	return {
		type: 'tool_arg_stream',
		offset: 0,
		data
	} as StreamEvent;
}

describe('buildArgStreamMap', () => {
	it('returns empty map for no events', () => {
		const m = buildArgStreamMap([]);
		expect(m.size).toBe(0);
	});

	it('initializes call entry on started phase', () => {
		const m = buildArgStreamMap([
			ev({ call_id: 'abc', phase: 'started', tool: 'code_exec', iteration: 1 })
		]);
		expect(m.get('abc')).toEqual({
			field: '',
			text: '',
			complete: false,
			tool: 'code_exec',
			iteration: 1
		});
	});

	it('accumulates delta text per call_id and preserves tool/iteration', () => {
		const m = buildArgStreamMap([
			ev({ call_id: 'abc', phase: 'started', tool: 'code_exec', iteration: 2 }),
			ev({ call_id: 'abc', phase: 'delta', field: 'code', text: 'hi ' }),
			ev({ call_id: 'abc', phase: 'delta', field: 'code', text: 'world' })
		]);
		expect(m.get('abc')).toEqual({
			field: 'code',
			text: 'hi world',
			complete: false,
			tool: 'code_exec',
			iteration: 2
		});
	});

	it('marks complete on completed phase', () => {
		const m = buildArgStreamMap([
			ev({ call_id: 'abc', phase: 'started', tool: 'code_exec', iteration: 1 }),
			ev({ call_id: 'abc', phase: 'delta', field: 'code', text: 'x' }),
			ev({ call_id: 'abc', phase: 'completed' })
		]);
		expect(m.get('abc')).toEqual({
			field: 'code',
			text: 'x',
			complete: true,
			tool: 'code_exec',
			iteration: 1
		});
	});

	it('keeps separate entries per call_id', () => {
		const m = buildArgStreamMap([
			ev({ call_id: 'a', phase: 'started' }),
			ev({ call_id: 'b', phase: 'started' }),
			ev({ call_id: 'a', phase: 'delta', field: 'code', text: 'AA' }),
			ev({ call_id: 'b', phase: 'delta', field: 'code', text: 'BB' })
		]);
		expect(m.get('a')?.text).toBe('AA');
		expect(m.get('b')?.text).toBe('BB');
	});

	it('ignores non-tool_arg_stream events', () => {
		const m = buildArgStreamMap([
			{ type: 'tool_status', offset: 0, data: { call_id: 'abc' } } as StreamEvent,
			ev({ call_id: 'abc', phase: 'started' })
		]);
		expect(m.size).toBe(1);
	});

	it('drops events with missing call_id', () => {
		const m = buildArgStreamMap([
			ev({ phase: 'delta', field: 'code', text: 'orphan' })
		]);
		expect(m.size).toBe(0);
	});
});
