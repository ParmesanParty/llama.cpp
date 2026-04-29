import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { MergedOrchestrationStore } from '$lib/stores/merged-orchestration.svelte';

interface DeferredFetch {
	resolveOk: (body: unknown) => void;
	resolveStatus: (status: number) => void;
	reject: (e: unknown) => void;
}

function buildDeferredFetchMock(): {
	fetchMock: ReturnType<typeof vi.fn>;
	latest: () => DeferredFetch;
} {
	let latest: DeferredFetch | null = null;
	const fetchMock = vi.fn().mockImplementation(() => {
		return new Promise<Response>((resolve, reject) => {
			latest = {
				resolveOk: (body) =>
					resolve(
						new Response(JSON.stringify(body), {
							status: 200,
							headers: { 'Content-Type': 'application/json' }
						})
					),
				resolveStatus: (status) => resolve(new Response('', { status })),
				reject
			};
		});
	});
	return {
		fetchMock,
		latest: () => {
			if (!latest) throw new Error('no fetch call yet');
			return latest;
		}
	};
}

const CAPABILITY_BODY = {
	merged_orchestration: {
		enabled: true,
		version: 1,
		endpoints: { sessions: '/api/sessions', tool_callback: '/api/tool-callback' },
		limits: {}
	}
};

describe('MergedOrchestrationStore — atomic reconcile', () => {
	let store: MergedOrchestrationStore;
	let fetchMock: ReturnType<typeof vi.fn>;
	let latest: () => DeferredFetch;
	let beaconCalls: string[];

	beforeEach(() => {
		({ fetchMock, latest } = buildDeferredFetchMock());
		vi.stubGlobal('fetch', fetchMock);
		beaconCalls = [];
		vi.stubGlobal('navigator', {
			sendBeacon: vi.fn().mockImplementation((url: string) => {
				beaconCalls.push(url);
				return true;
			})
		});
		store = new MergedOrchestrationStore();
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.clearAllMocks();
	});

	async function probeAndRegisterInitial(
		sessionId = 'sid-1',
		sessionToken = 'tok-1'
	): Promise<void> {
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;

		const regPromise = store.registerSession();
		latest().resolveOk({ session_id: sessionId, session_token: sessionToken });
		await regPromise;
	}

	it('reconcile preserves OLD session id while new registration is in flight', async () => {
		await probeAndRegisterInitial('old-sid', 'old-tok');
		expect(store.sessionId).toBe('old-sid');
		expect(store.sessionToken).toBe('old-tok');

		const reconcilePromise = store.reconcile();

		// Synchronously after kick-off (before the fetch resolves), the OLD
		// session is still in place.
		expect(store.sessionId).toBe('old-sid');
		expect(store.sessionToken).toBe('old-tok');

		latest().resolveOk({ session_id: 'new-sid', session_token: 'new-tok' });
		const replaced = await reconcilePromise;

		expect(replaced).toBe(true);
		expect(store.sessionId).toBe('new-sid');
		expect(store.sessionToken).toBe('new-tok');
	});

	it('successful reconcile fires close-beacon for the OLD session AFTER the swap', async () => {
		await probeAndRegisterInitial('old-sid', 'old-tok');
		const reconcilePromise = store.reconcile();
		latest().resolveOk({ session_id: 'new-sid', session_token: 'new-tok' });
		await reconcilePromise;

		expect(beaconCalls.length).toBe(1);
		expect(beaconCalls[0]).toContain('/api/sessions/old-sid/close');
		expect(beaconCalls[0]).toContain('token=old-tok');
		expect(beaconCalls[0]).not.toContain('new-sid');
	});

	it('failed reconcile (HTTP 5xx) preserves the OLD session and returns false', async () => {
		await probeAndRegisterInitial('old-sid', 'old-tok');

		const reconcilePromise = store.reconcile();
		latest().resolveStatus(500);
		const replaced = await reconcilePromise;

		expect(replaced).toBe(false);
		expect(store.sessionId).toBe('old-sid');
		expect(store.sessionToken).toBe('old-tok');
		expect(beaconCalls.length).toBe(0);
	});

	it('failed reconcile (network error) preserves the OLD session and returns false', async () => {
		await probeAndRegisterInitial('old-sid', 'old-tok');

		const reconcilePromise = store.reconcile();
		latest().reject(new Error('network down'));
		const replaced = await reconcilePromise;

		expect(replaced).toBe(false);
		expect(store.sessionId).toBe('old-sid');
		expect(beaconCalls.length).toBe(0);
	});

	it('reconcile with no prior session: registers a new one, no beacon', async () => {
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;

		const reconcilePromise = store.reconcile();
		latest().resolveOk({ session_id: 'first-sid', session_token: 'first-tok' });
		const replaced = await reconcilePromise;

		expect(replaced).toBe(true);
		expect(store.sessionId).toBe('first-sid');
		expect(beaconCalls.length).toBe(0);
	});

	it('reconcile is a no-op (returns false) when capability is disabled', async () => {
		const probePromise = store.probeCapability();
		latest().resolveOk({ merged_orchestration: { enabled: false } });
		await probePromise;

		const replaced = await store.reconcile();
		expect(replaced).toBe(false);
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});

	it('reconcile-during-reconcile: coalesces, beacons OLD exactly once, never beacons NEW', async () => {
		await probeAndRegisterInitial('old-sid', 'old-tok');
		fetchMock.mockClear();
		// fetchMock.mockClear() also clears the cached `latest` reference;
		// re-bind it via the helper exposed by buildDeferredFetchMock.
		beaconCalls.length = 0;

		// Two reconciles fired before any fetch resolves — second coalesces on
		// _registrationInFlight. Both should resolve to the same SessionInfo.
		const r1 = store.reconcile();
		const r2 = store.reconcile();

		// Coalesced: only one /api/sessions fetch issued.
		expect(fetchMock).toHaveBeenCalledTimes(1);

		latest().resolveOk({ session_id: 'new-sid', session_token: 'new-tok' });
		const [b1, b2] = await Promise.all([r1, r2]);

		expect(b1).toBe(true);
		expect(b2).toBe(true);
		expect(store.sessionId).toBe('new-sid');

		// Beacon fires exactly once for the OLD session. The reference-equality
		// guard prevents the second reconcile (whose `previous` IS the
		// just-installed `new`) from beaconing the just-minted session.
		expect(beaconCalls.length).toBe(1);
		expect(beaconCalls[0]).toContain('/api/sessions/old-sid/close');
		expect(beaconCalls[0]).not.toContain('new-sid');
	});

	it('coalesced registerSession+reconcile does NOT beacon the just-minted session', async () => {
		// Probe enabled, no prior session.
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;

		// Cold-start: registerSession fires, fetch in flight.
		const regPromise = store.registerSession();
		// MCP-set-change effect fires reconcile while registration is still in flight.
		const reconcilePromise = store.reconcile();

		// Single fetch shared by both.
		expect(fetchMock).toHaveBeenCalledTimes(2); // 1 probe + 1 sessions

		latest().resolveOk({ session_id: 'shared', session_token: 'sharedtok' });
		await Promise.all([regPromise, reconcilePromise]);

		expect(store.sessionId).toBe('shared');
		// CRITICAL: the just-minted session must NOT be beacon-closed.
		expect(beaconCalls.length).toBe(0);
	});
});

describe('MergedOrchestrationStore — waitUntilReady', () => {
	let store: MergedOrchestrationStore;
	let fetchMock: ReturnType<typeof vi.fn>;
	let latest: () => DeferredFetch;

	beforeEach(() => {
		({ fetchMock, latest } = buildDeferredFetchMock());
		vi.stubGlobal('fetch', fetchMock);
		vi.stubGlobal('navigator', { sendBeacon: vi.fn().mockReturnValue(true) });
		store = new MergedOrchestrationStore();
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.clearAllMocks();
	});

	it('resolves immediately when no registration is in flight', async () => {
		const start = performance.now();
		await store.waitUntilReady();
		expect(performance.now() - start).toBeLessThan(50);
	});

	it('awaits an in-flight registerSession (cold-start race)', async () => {
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;

		const regPromise = store.registerSession();

		let resolved = false;
		const waitPromise = store.waitUntilReady().then(() => {
			resolved = true;
		});

		await Promise.resolve();
		expect(resolved).toBe(false);

		latest().resolveOk({ session_id: 'sid-cold', session_token: 'tok-cold' });
		await regPromise;
		await waitPromise;

		expect(resolved).toBe(true);
		expect(store.sessionId).toBe('sid-cold');
	});

	it('awaits an in-flight reconcile (chains through new-session registration)', async () => {
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;
		const regPromise = store.registerSession();
		latest().resolveOk({ session_id: 'old', session_token: 'oldtok' });
		await regPromise;

		const reconcilePromise = store.reconcile();

		let resolved = false;
		const waitPromise = store.waitUntilReady().then(() => {
			resolved = true;
		});
		await Promise.resolve();
		expect(resolved).toBe(false);

		latest().resolveOk({ session_id: 'new', session_token: 'newtok' });
		await reconcilePromise;
		await waitPromise;

		expect(resolved).toBe(true);
		expect(store.sessionId).toBe('new');
	});

	it('concurrent registerSession() calls share one in-flight registration', async () => {
		const probePromise = store.probeCapability();
		latest().resolveOk(CAPABILITY_BODY);
		await probePromise;
		fetchMock.mockClear();

		const p1 = store.registerSession();
		const p2 = store.registerSession();

		expect(fetchMock).toHaveBeenCalledTimes(1);

		latest().resolveOk({ session_id: 'shared', session_token: 'sharedtok' });
		await Promise.all([p1, p2]);

		expect(store.sessionId).toBe('shared');
		expect(fetchMock).toHaveBeenCalledTimes(1);
	});
});
