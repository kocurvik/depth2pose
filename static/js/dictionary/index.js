import { t } from './dict.js';

/**
 * Applies text from the dictionary to all elements carrying a `data-i18n` attribute.
 * - Should be called once at startup, or again after dynamic DOM changes.
 */
export function applyStaticTexts() {
	document.querySelectorAll('[data-i18n]').forEach((el) => {
		const key = el.getAttribute('data-i18n');
		const entry = t(key);

		if (!entry) return;
		const attr = el.getAttribute('data-i18n-attr');

		if (attr) el.setAttribute(attr, entry.label);
		else {
			el.innerHTML = entry.label;
			setOptionTitle(el, entry.description);
		}
	});
}

/*
 * Dictionary helper functions for simplified access to labels and descriptions, with optional fallbacks.
 */

/** Get just the label for a key, with an optional fallback. */
export function tLabel(key, params = {}) {
	const entry = t(key);
	if (!entry) return key;

	if (!params) return entry.label;
	return entry.label.replace(/\{(\w+)\}/g, (_, name) => { return params[name] ?? `{${name}}`; });
}


/** Get the description for a key (for tooltips), or null. */
export function tDesc(key, params = {}) {
	const entry = t(key);
	if (!entry) return null;

	if (!params) return entry.description ?? null;
	return entry.description?.replace(/\{(\w+)\}/g, (_, name) => { return params[name] ?? `{${name}}`; });
}

/** Generate a dictionary key for a CSV value. */
export function csvValue(kind, rawValue) {
	return `table.value.${kind}.${rawValue}`;
}

/** Look up a display label for a raw CSV value. */
export function csvValueLabel(kind, rawValue) {
	if (typeof rawValue === "number") {
		if (Number.isInteger(rawValue)) return rawValue;
		return rawValue.toFixed(2);
	}
	return t(csvValue(kind, rawValue))?.label ?? rawValue;
}

/** Look up a description for a raw CSV value (for tooltips). */
export function csvValueDesc(kind, rawValue) {
	return t(csvValue(kind, rawValue))?.description ?? null;
}

/*
 * Helper functions for building HTML with dictionary content, including tooltips.
 */

/** Get the title attribute for an <option> based on its dictionary key and parameters. */
export function getTitleAttr(key, params = {}) {
	const desc = tDesc(key, params);
	return desc ? ` title="${desc}"` : '';
}

/** Set `title` on an <option> if description is truthy. */
export function setOptionTitle(optionEl, description) {
	if (description) optionEl.setAttribute('title', description);
}