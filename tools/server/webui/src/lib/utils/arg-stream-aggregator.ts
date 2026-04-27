import type { StreamEvent } from '$lib/types';

export interface ArgStreamState {
	field: string;
	text: string;
	complete: boolean;
	tool?: string;
	iteration?: number;
}

/**
 * Reduces a sequence of stream events into a per-call_id map of in-flight
 * argument decoding state. The orchestrator emits these as the model writes
 * a tool's primary string argument (e.g. code_exec's `code`); the UI uses
 * the result to render a "writing" pill before the tool actually executes.
 *
 * `tool` and `iteration` are captured from the `started` event so the chip
 * renderer can place an argStream-only chip in the correct iteration group
 * before any tool_status event has fired for that call_id.
 *
 * Pure: single pass, no I/O, no side effects.
 */
export function buildArgStreamMap(events: StreamEvent[]): Map<string, ArgStreamState> {
	const map = new Map<string, ArgStreamState>();
	for (const event of events) {
		if (event.type !== 'tool_arg_stream') continue;
		const d = event.data as {
			call_id?: string;
			phase?: string;
			field?: string;
			text?: string;
			tool?: string;
			iteration?: number;
		};
		if (!d.call_id) continue;
		const cur = map.get(d.call_id) ?? { field: '', text: '', complete: false };
		if (d.phase === 'started') {
			map.set(d.call_id, {
				field: '',
				text: '',
				complete: false,
				tool: d.tool,
				iteration: d.iteration
			});
		} else if (d.phase === 'delta') {
			map.set(d.call_id, {
				field: d.field ?? cur.field,
				text: cur.text + (d.text ?? ''),
				complete: cur.complete,
				tool: cur.tool,
				iteration: cur.iteration
			});
		} else if (d.phase === 'completed') {
			map.set(d.call_id, { ...cur, complete: true });
		}
	}
	return map;
}
