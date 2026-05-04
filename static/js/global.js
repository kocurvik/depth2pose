import { setOptionTitle, tDesc, tLabel } from "./dictionary/index.js";

/** Fetch a JSON file from the given URL and return its parsed contents. */
export async function fetchJson(url) {
	const response = await fetch(url);

	if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
	return response.json();
}

/** Attach toggle functionality to a card, allowing it to collapse/expand when the button is clicked. */
export function attachCardToggle(cardId, buttonId, contentId, showLang, hideLang) {
	const card = document.getElementById(cardId);
	const button = document.getElementById(buttonId);
	const content = document.getElementById(contentId);
	if (!card || !button || !content || button.dataset.bound) return;

	button.dataset.bound = 'true';
	button.addEventListener('click', () => {
		const collapsed = card.classList.toggle('is-collapsed');
		
		button.setAttribute('aria-expanded', String(!collapsed));
		button.textContent = collapsed ? tLabel(showLang.key, showLang.params) : tLabel(hideLang.key, hideLang.params);
		setOptionTitle(button, collapsed ? tDesc(showLang.key, showLang.params) : tDesc(hideLang.key, hideLang.params));
	});
}


/** Check if an MDE name represents a ground-truth/oracle result. */
export function isGtMde(value) {
	const normalized = normalizeForSearch(value);
	return normalized === 'gt' || normalized === 'ground-truth' || normalized === 'ground_truth' || normalized === 'none';
}

/** Check if an MDE name indicates that it is a calibrated method, based on the presence of 'Calib' in the name. */
export function isCalibMde(mde) {
	return normalizeForSearch(mde).includes('calib');
}

/** Extract the MDE family key by taking the prefix before any '-' character and removing 'Calib', used for grouping related MDEs together. */
export function getMdeFamilyKey(mde) {
	const prefix = String(mde ?? '').split('-')[0];
	return prefix.replaceAll('Calib', '');
}

/** Check if a solver name indicates that it is a 'ro' variant, based on the presence of the '_ro' suffix. */
export function isRoSolver(solver) {
	return normalizeForSearch(solver).endsWith('_ro');
}

/** Extract the base solver name by removing any '_ro' suffix, used for grouping related solvers together. */
export function baseSolverName(solver) {
	return String(solver ?? '').replace(/_ro$/, '');
}


/** Sort strings by the same natural ordering used across controls and cards. */
export function sortByName(a, b) {
	return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
}


/** Normalize a value for case-insensitive text matching. */
export function normalizeForSearch(value) {
	return String(value ?? '').trim().toLowerCase();
}

/** Utility function to check if a value should be parsed as a number. */
export function isNumericValue(value) {
	if (value === '' || value === null || value === undefined) return false;
	return !Number.isNaN(Number(value));
}