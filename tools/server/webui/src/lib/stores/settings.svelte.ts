/**
 * settingsStore - Application configuration and theme management
 *
 * This store manages all application settings including AI model parameters, UI preferences,
 * and theme configuration. It provides persistent storage through localStorage with reactive
 * state management using Svelte 5 runes.
 *
 * **Architecture & Relationships:**
 * - **settingsStore** (this class): Configuration state management
 *   - Manages AI model parameters (temperature, max tokens, etc.)
 *   - Handles theme switching and persistence
 *   - Provides localStorage synchronization
 *   - Offers reactive configuration access
 *
 * - **ChatService**: Reads model parameters for API requests
 * - **UI Components**: Subscribe to theme and configuration changes
 *
 * **Key Features:**
 * - **Model Parameters**: Temperature, max tokens, top-p, top-k, repeat penalty
 * - **Theme Management**: Auto, light, dark theme switching
 * - **Persistence**: Automatic localStorage synchronization
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 * - **Default Handling**: Graceful fallback to defaults for missing settings
 * - **Batch Updates**: Efficient multi-setting updates
 * - **Reset Functionality**: Restore defaults for individual or all settings
 *
 * **Configuration Categories:**
 * - Generation parameters (temperature, tokens, sampling)
 * - UI preferences (theme, display options)
 * - System settings (model selection, prompts)
 * - Advanced options (seed, penalties, context handling)
 */

import { browser } from '$app/environment';
import {
	CONFIG_LOCALSTORAGE_KEY,
	SETTING_CONFIG_DEFAULT,
	USER_OVERRIDES_LOCALSTORAGE_KEY
} from '$lib/constants';
import { IsMobile } from '$lib/hooks/is-mobile.svelte';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { serverStore } from '$lib/stores/server.svelte';
import {
	configToParameterRecord,
	normalizeFloatingPoint,
	getConfigValue,
	setConfigValue
} from '$lib/utils';

class SettingsStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	config = $state<SettingsConfigType>({ ...SETTING_CONFIG_DEFAULT });
	theme = $state<string>('auto');
	isInitialized = $state(false);
	userOverrides = $state<Set<string>>(new Set());

	// Thinking-mode override state
	private _thinkingOverrides = $state<Record<string, number>>({});
	private _thinkingUserOverrides = $state<Set<string>>(new Set());
	private static readonly THINKING_OVERRIDES_KEY = 'LlamacppWebui.thinkingOverrides';
	private static readonly THINKING_USER_OVERRIDES_KEY = 'LlamacppWebui.thinkingUserOverrides';

	/**
	 *
	 *
	 * Utilities (private helpers)
	 *
	 *
	 */

	/**
	 * Helper method to get server defaults with null safety
	 * Centralizes the pattern of getting and extracting server defaults
	 */
	private getServerDefaults(): Record<string, string | number | boolean> {
		const serverParams = serverStore.defaultParams;
		const webuiSettings = serverStore.webuiSettings;
		return ParameterSyncService.extractServerDefaults(serverParams, webuiSettings);
	}

	constructor() {
		if (browser) {
			this.initialize();

			// Load thinking overrides from localStorage
			try {
				const saved = localStorage.getItem(SettingsStore.THINKING_OVERRIDES_KEY);
				if (saved) this._thinkingOverrides = JSON.parse(saved);
				const savedUserOv = localStorage.getItem(SettingsStore.THINKING_USER_OVERRIDES_KEY);
				if (savedUserOv) this._thinkingUserOverrides = new Set(JSON.parse(savedUserOv));
			} catch {
				// Ignore parse errors — use defaults
			}
		}
	}

	/**
	 *
	 *
	 * Lifecycle
	 *
	 *
	 */

	/**
	 * Initialize the settings store by loading from localStorage
	 */
	initialize() {
		try {
			this.loadConfig();
			this.loadTheme();
			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize settings store:', error);
		}
	}

	/**
	 * Load configuration from localStorage
	 * Returns default values for missing keys to prevent breaking changes
	 */
	private loadConfig() {
		if (!browser) return;

		try {
			const storedConfigRaw = localStorage.getItem(CONFIG_LOCALSTORAGE_KEY);
			const savedVal = JSON.parse(storedConfigRaw || '{}');

			// Merge with defaults to prevent breaking changes
			this.config = {
				...SETTING_CONFIG_DEFAULT,
				...savedVal
			};

			// Default sendOnEnter to false on mobile when the user has no saved preference
			if (!('sendOnEnter' in savedVal)) {
				if (new IsMobile().current) {
					this.config.sendOnEnter = false;
				}
			}

			// Load user overrides
			const savedOverrides = JSON.parse(
				localStorage.getItem(USER_OVERRIDES_LOCALSTORAGE_KEY) || '[]'
			);
			this.userOverrides = new Set(savedOverrides);
		} catch (error) {
			console.warn('Failed to parse config from localStorage, using defaults:', error);
			this.config = { ...SETTING_CONFIG_DEFAULT };
			this.userOverrides = new Set();
		}
	}

	/**
	 * Load theme from localStorage
	 */
	private loadTheme() {
		if (!browser) return;

		this.theme = localStorage.getItem('theme') || 'auto';
	}
	/**
	 *
	 *
	 * Config Updates
	 *
	 *
	 */

	/**
	 * Update a specific configuration setting
	 * @param key - The configuration key to update
	 * @param value - The new value for the configuration key
	 */
	updateConfig<K extends keyof SettingsConfigType>(key: K, value: SettingsConfigType[K]): void {
		this.config[key] = value;

		if (ParameterSyncService.canSyncParameter(key as string)) {
			const propsDefaults = this.getServerDefaults();
			const propsDefault = propsDefaults[key as string];

			if (propsDefault !== undefined) {
				const normalizedValue = normalizeFloatingPoint(value);
				const normalizedDefault = normalizeFloatingPoint(propsDefault);

				if (normalizedValue === normalizedDefault) {
					this.userOverrides.delete(key as string);
				} else {
					this.userOverrides.add(key as string);
				}
			}
		}

		this.saveConfig();
	}

	/**
	 * Update multiple configuration settings at once
	 * @param updates - Object containing the configuration updates
	 */
	updateMultipleConfig(updates: Partial<SettingsConfigType>) {
		Object.assign(this.config, updates);

		const propsDefaults = this.getServerDefaults();

		for (const [key, value] of Object.entries(updates)) {
			if (ParameterSyncService.canSyncParameter(key)) {
				const propsDefault = propsDefaults[key];

				if (propsDefault !== undefined) {
					const normalizedValue = normalizeFloatingPoint(value);
					const normalizedDefault = normalizeFloatingPoint(propsDefault);

					if (normalizedValue === normalizedDefault) {
						this.userOverrides.delete(key);
					} else {
						this.userOverrides.add(key);
					}
				}
			}
		}

		this.saveConfig();
	}

	/**
	 * Save the current configuration to localStorage
	 */
	private saveConfig() {
		if (!browser) return;

		try {
			localStorage.setItem(CONFIG_LOCALSTORAGE_KEY, JSON.stringify(this.config));

			localStorage.setItem(
				USER_OVERRIDES_LOCALSTORAGE_KEY,
				JSON.stringify(Array.from(this.userOverrides))
			);
		} catch (error) {
			console.error('Failed to save config to localStorage:', error);
		}
	}

	/**
	 * Update the theme setting
	 * @param newTheme - The new theme value
	 */
	updateTheme(newTheme: string) {
		this.theme = newTheme;
		this.saveTheme();
	}

	/**
	 * Save the current theme to localStorage
	 */
	private saveTheme() {
		if (!browser) return;

		try {
			if (this.theme === 'auto') {
				localStorage.removeItem('theme');
			} else {
				localStorage.setItem('theme', this.theme);
			}
		} catch (error) {
			console.error('Failed to save theme to localStorage:', error);
		}
	}

	/**
	 *
	 *
	 * Reset
	 *
	 *
	 */

	/**
	 * Reset configuration to defaults
	 */
	resetConfig() {
		this.config = { ...SETTING_CONFIG_DEFAULT };
		this.saveConfig();
	}

	/**
	 * Reset theme to auto
	 */
	resetTheme() {
		this.theme = 'auto';
		this.saveTheme();
	}

	/**
	 * Reset all settings to defaults
	 */
	resetAll() {
		this.resetConfig();
		this.resetTheme();
	}

	/**
	 * Reset a parameter to server default (or webui default if no server default)
	 */
	resetParameterToServerDefault(key: string): void {
		const serverDefaults = this.getServerDefaults();
		const webuiSettings = serverStore.webuiSettings;

		if (webuiSettings && key in webuiSettings) {
			// UI setting from admin config: write actual value
			setConfigValue(this.config, key, webuiSettings[key]);
		} else if (serverDefaults[key] !== undefined) {
			// sampling param known by server: clear it, let server decide
			setConfigValue(this.config, key, '');
		} else if (key in SETTING_CONFIG_DEFAULT) {
			setConfigValue(this.config, key, getConfigValue(SETTING_CONFIG_DEFAULT, key));
		}

		this.userOverrides.delete(key);
		this.saveConfig();
	}

	/**
	 *
	 *
	 * Thinking Overrides
	 *
	 *
	 */

	/** Server-provided thinking overrides (from /props). */
	get thinkingOverrides(): Record<string, number> {
		return this._thinkingOverrides;
	}

	/** Whether any thinking overrides are configured. */
	get hasThinkingOverrides(): boolean {
		return Object.keys(this._thinkingOverrides).length > 0;
	}

	/** Get the effective thinking value for a param (user override or server default). */
	getThinkingValue(param: string): number | undefined {
		return this._thinkingOverrides[param];
	}

	/** Whether the user has customized a specific thinking param. */
	isThinkingUserOverride(param: string): boolean {
		return this._thinkingUserOverrides.has(param);
	}

	/** Set a user-customized thinking override value. */
	setThinkingOverride(param: string, value: number): void {
		this._thinkingOverrides[param] = value;
		this._thinkingUserOverrides.add(param);
		this.saveThinkingOverrides();
	}

	/** Reset a thinking override to the server default. */
	resetThinkingOverride(param: string, serverValue: number): void {
		this._thinkingOverrides[param] = serverValue;
		this._thinkingUserOverrides.delete(param);
		this.saveThinkingOverrides();
	}

	/** Sync thinking overrides from /props server response.
	 *  Uses snapshots of reactive state to avoid triggering $effect loops
	 *  when called from within a reactive context (e.g. +layout.svelte).
	 */
	syncThinkingOverrides(serverOverrides: Record<string, number>): void {
		// Snapshot current state to avoid reading $state during merge
		const currentOverrides = { ...this._thinkingOverrides };
		const currentUserOv = new Set(this._thinkingUserOverrides);

		const merged: Record<string, number> = {};
		for (const [key, value] of Object.entries(serverOverrides)) {
			if (currentUserOv.has(key) && key in currentOverrides) {
				merged[key] = currentOverrides[key];
			} else {
				merged[key] = value;
			}
		}

		// Prune user overrides for keys no longer in server set
		const prunedUserOv = new Set(currentUserOv);
		for (const key of currentUserOv) {
			if (!(key in serverOverrides)) {
				prunedUserOv.delete(key);
			}
		}

		// Single write to each $state field (no read-after-write)
		this._thinkingOverrides = merged;
		this._thinkingUserOverrides = prunedUserOv;
		this.saveThinkingOverrides();
	}

	private saveThinkingOverrides(): void {
		if (!browser) return;
		localStorage.setItem(
			SettingsStore.THINKING_OVERRIDES_KEY,
			JSON.stringify(this._thinkingOverrides)
		);
		localStorage.setItem(
			SettingsStore.THINKING_USER_OVERRIDES_KEY,
			JSON.stringify([...this._thinkingUserOverrides])
		);
	}

	/**
	 *
	 *
	 * Server Sync
	 *
	 *
	 */

	/**
	 * Initialize settings with props defaults when server properties are first loaded
	 * This sets up the default values from /props endpoint
	 */
	syncWithServerDefaults(): void {
		const propsDefaults = this.getServerDefaults();
		if (Object.keys(propsDefaults).length === 0) return;

		for (const [key, propsValue] of Object.entries(propsDefaults)) {
			const currentValue = getConfigValue(this.config, key);

			const normalizedCurrent = normalizeFloatingPoint(currentValue);
			const normalizedDefault = normalizeFloatingPoint(propsValue);

			// if user value matches server, it's not a real override
			if (normalizedCurrent === normalizedDefault) {
				this.userOverrides.delete(key);
			}
		}

		// webui settings need actual values in config (no placeholder mechanism),
		// so write them for non-overridden keys
		const webuiSettings = serverStore.webuiSettings;
		if (webuiSettings) {
			for (const [key, value] of Object.entries(webuiSettings)) {
				if (!this.userOverrides.has(key) && value !== undefined) {
					setConfigValue(this.config, key, value);
				}
			}
		}

		this.saveConfig();
		console.log('Settings initialized with props defaults:', propsDefaults);
		console.log('Current user overrides after sync:', Array.from(this.userOverrides));

		// Sync thinking overrides
		const thinkingOv = ParameterSyncService.extractThinkingOverrides(serverStore.props);
		this.syncThinkingOverrides(thinkingOv);
	}

	/**
	 * Reset all parameters to their default values (from props)
	 * This is used by the "Reset to Default" functionality
	 * Prioritizes server defaults from /props, falls back to webui defaults
	 */
	forceSyncWithServerDefaults(): void {
		const propsDefaults = this.getServerDefaults();
		const webuiSettings = serverStore.webuiSettings;

		for (const key of ParameterSyncService.getSyncableParameterKeys()) {
			if (webuiSettings && key in webuiSettings) {
				// UI setting from admin config: write actual value
				setConfigValue(this.config, key, webuiSettings[key]);
			} else if (propsDefaults[key] !== undefined) {
				// sampling param: clear it, let server decide
				setConfigValue(this.config, key, '');
			} else if (key in SETTING_CONFIG_DEFAULT) {
				setConfigValue(this.config, key, getConfigValue(SETTING_CONFIG_DEFAULT, key));
			}

			this.userOverrides.delete(key);
		}

		this.saveConfig();
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	/**
	 * Get a specific configuration value
	 * @param key - The configuration key to get
	 * @returns The configuration value
	 */
	getConfig<K extends keyof SettingsConfigType>(key: K): SettingsConfigType[K] {
		return this.config[key];
	}

	/**
	 * Get the entire configuration object
	 * @returns The complete configuration object
	 */
	getAllConfig(): SettingsConfigType {
		return { ...this.config };
	}

	canSyncParameter(key: string): boolean {
		return ParameterSyncService.canSyncParameter(key);
	}

	/**
	 * Get parameter information including source for a specific parameter
	 */
	getParameterInfo(key: string) {
		const propsDefaults = this.getServerDefaults();
		const currentValue = getConfigValue(this.config, key);

		return ParameterSyncService.getParameterInfo(
			key,
			currentValue ?? '',
			propsDefaults,
			this.userOverrides
		);
	}

	/**
	 * Get diff between current settings and server defaults
	 */
	getParameterDiff() {
		const serverDefaults = this.getServerDefaults();
		if (Object.keys(serverDefaults).length === 0) return {};

		const configAsRecord = configToParameterRecord(
			this.config,
			ParameterSyncService.getSyncableParameterKeys()
		);

		return ParameterSyncService.createParameterDiff(configAsRecord, serverDefaults);
	}

	/**
	 * Clear all user overrides (for debugging)
	 */
	clearAllUserOverrides(): void {
		this.userOverrides.clear();
		this.saveConfig();
		console.log('Cleared all user overrides');
	}
}

export const settingsStore = new SettingsStore();

export const config = () => settingsStore.config;
export const theme = () => settingsStore.theme;
export const isInitialized = () => settingsStore.isInitialized;
