import { IMG_EXAMPLES_CONTENTS_URL } from '../api-config.js';
import { fetchJson, sortByName, escapeHtml } from '../global.js';

import { datasetAssetUrl, getBestResult, normalizePairKey } from './shared.js';

const DATASET_MANIFEST_CACHE = new Map();
const DATASET_EXAMPLES_CACHE = new Map();


/* List dataset folders directly from benchmark/d2p_examples and render the chooser. */
export async function loadAvailableDatasets(state, controller) {
	resetDatasetState(state);

	controller.renderPanelShell(
		'Image Examples',
		'Choose a dataset to browse its all example image pairs.',
		'<div class="example-loading">Loading example datasets…</div>'
	);

	try {
		const datasets = await listExampleDatasets();
		state.availableDatasets = datasets;

		if (!datasets.length) {
			controller.renderPanelShell(
				'Image Examples',
				'No dataset folders were found in d2p_examples.',
				'<div class="example-empty-state">No example datasets are currently available.</div>'
			);
			return;
		}

		state.datasetSummaries = await loadAllDatasetSummaries(state.availableDatasets);
		renderDatasetChooserView(state, controller);
	}
	catch (error) {
		console.error(error);
		controller.renderPanelShell(
			'Image Examples',
			'Example datasets are loaded independently from the benchmark table.',
			'<div class="example-empty-state">Failed to list dataset folders from d2p_examples.</div>'
		);
	}
}

/* Load all detailed examples for one dataset and cache the parsed example objects. */
export async function loadDetailedExamples(dataset) {
	if (DATASET_EXAMPLES_CACHE.has(dataset)) return DATASET_EXAMPLES_CACHE.get(dataset);

	const promise = (async () => {
		const manifest = await loadDatasetManifest(dataset);
		const entries = Object.entries(manifest || {});

		return Promise.all(entries.map(async ([pairKey, baseEntry]) => {
			const resultPath = baseEntry.result_path || `results/${normalizePairKey(pairKey)}.json`;
			const details = await fetchJson(datasetAssetUrl(dataset, resultPath));
			const selectedResult = getBestResult(details);

			return {
				pairKey, details, selectedResult,
				rgb1Url: datasetAssetUrl(dataset, baseEntry.exported_rgb_image1_path),
				rgb2Url: datasetAssetUrl(dataset, baseEntry.exported_rgb_image2_path)
			};
		}));
	})();

	DATASET_EXAMPLES_CACHE.set(dataset, promise);
	return promise;
}

/* Return folder names from the d2p_examples directory. */
async function listExampleDatasets() {
	const entries = await fetchJson(IMG_EXAMPLES_CONTENTS_URL);
	
	if (!Array.isArray(entries)) return [];
	return entries
		.filter((entry) => entry?.type === 'dir' && entry.name)
		.map((entry) => entry.name)
		.sort(sortByName);
}

/* Load lightweight manifest summaries for every listed dataset folder. */
async function loadAllDatasetSummaries(datasets) {
	const summaries = await Promise.all(datasets.map(async (datasetName) => {
		try {
			const manifest = await loadDatasetManifest(datasetName);
			const count = manifest && typeof manifest === 'object' ? Object.keys(manifest).length : 0;

			return { dataset: datasetName, count };
		}
		catch {
			return { dataset: datasetName, count: 0 };
		}
	}));

	return summaries.sort((a, b) => sortByName(a.dataset, b.dataset));
}

/* Load the dataset manifest for one dataset and cache the result. */
async function loadDatasetManifest(dataset) {
	if (DATASET_MANIFEST_CACHE.has(dataset)) return DATASET_MANIFEST_CACHE.get(dataset);

	const promise = fetchJson(datasetAssetUrl(dataset, `${dataset}_examples.json`));
	DATASET_MANIFEST_CACHE.set(dataset, promise);
	return promise;
}


/* Open the pair-list workflow for one concrete dataset. */
export async function openDatasetPairList(datasetName, state, controller) {
	state.currentDatasetName = datasetName;
	state.view = 'pairs';

	controller.renderPanelShell(
		'Image Examples',
		`Dataset: ${datasetName}`,
		`<div class="example-loading">Loading image pairs for ${escapeHtml(datasetName)}…</div>`,
		{
			showBackButton: true,
			backLabel: 'Back to datasets',
			onBack: () => {
				renderDatasetChooserView(state, controller);
				controller.focusExamplesViewer();
			}
		}
	);
	controller.focusExamplesViewer();

	try {
		const examples = await loadDetailedExamples(datasetName);
		state.currentExamples = examples;

		if (!examples.length) {
			controller.renderPanelShell(
				'Image Examples',
				`Dataset: ${datasetName}`,
				`<div class="example-empty-state">No image pairs are currently available for dataset ${escapeHtml(datasetName)}.</div>`,
				{
					showBackButton: true,
					backLabel: 'Back to datasets',
					onBack: () => {
						renderDatasetChooserView(state, controller);
						controller.focusExamplesViewer();
					}
				}
			);
			return;
		}

		controller.renderPairListView();
	}
	catch (error) {
		console.error(error);
		controller.renderPanelShell(
			'Image Examples',
			`Dataset: ${datasetName}`,
			`<div class="example-empty-state">Failed to load image pairs for dataset ${escapeHtml(datasetName)}.</div>`,
			{
				showBackButton: true,
				backLabel: 'Back to datasets',
				onBack: () => {
					renderDatasetChooserView(state, controller);
					controller.focusExamplesViewer();
				}
			}
		);
	}
}

/* Render the dataset chooser. */
export function renderDatasetChooserView(state, controller) {
	state.view = 'datasets';
	const summaries = state.datasetSummaries.filter((item) => item.count > 0);

	controller.renderPanelShell(
		'Image Examples',
		'Choose a dataset to browse its all example image pairs.',
		`
			<p class="muted-note mb-4">Select a dataset first, then click any pair to inspect its depth results.</p>
			<div class="dataset-summary-grid">
				${summaries.map((summary) => renderDatasetSummaryCard(summary)).join('')}
			</div>
		`,
		{ showBackButton: false }
	);
}

/* Render one dataset summary card. */
function renderDatasetSummaryCard(summary) {
	return `
		<article class="dataset-summary-card">
			<div>
				<h4 class="title is-5 mb-1">${escapeHtml(summary.dataset)}</h4>
				<p class="dataset-summary-subtitle mb-0">${summary.count} image pairs available</p>
			</div>

			<div class="dataset-summary-actions">
				<button class="button is-link is-light" type="button" data-open-dataset="${escapeHtml(summary.dataset)}">
					Open pairs
				</button>
			</div>
		</article>
	`;
}


/* Reset all dataset-dependent state before a fresh load. */
function resetDatasetState(state) {
	state.availableDatasets = [];
	state.datasetSummaries = [];
	state.currentDatasetName = null;
	state.currentExamples = [];
	state.view = 'datasets';
}