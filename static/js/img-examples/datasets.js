import { IMG_EXAMPLES_CONTENTS_URL } from '../api-config.js';
import { fetchJson, sortByName } from '../global.js';
import { datasetAssetUrl, getBestResult, normalizePairKey } from './shared.js';
import { tLabel, csvValue, csvValueLabel, getTitleAttr } from '../dictionary/index.js';

const DATASET_MANIFEST_CACHE = new Map();
const DATASET_EXAMPLES_CACHE = new Map();


/** List dataset folders directly from benchmark/d2p_examples and render the chooser. */
export async function loadAvailableDatasets(state, controller) {
	resetDatasetState(state);

	controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.choose-dataset' },
		`<div class="example-loading" ${getTitleAttr('examples.loading.datasets')}>
			${tLabel('examples.loading.datasets')}
		</div>`
	);

	try {
		const datasets = await listExampleDatasets();
		state.availableDatasets = datasets;

		if (!datasets.length) {
			controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.choose-dataset' },
				`<div class="example-empty-state" ${getTitleAttr('examples.empty.datasets')}>
					${tLabel('examples.empty.datasets')}
				</div>`
			);
			return;
		}

		state.datasetSummaries = await loadAllDatasetSummaries(state.availableDatasets);
		renderDatasetChooserView(state, controller);
	}
	catch (error) {
		console.error(error);
		controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.choose-dataset' },
			`<div class="example-empty-state" ${getTitleAttr('examples.failed.datasets')}>
				${tLabel('examples.failed.datasets')}
			</div>`
		);
	}
}

/** Load all detailed examples for one dataset and cache the parsed example objects. */
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

/** Return folder names from the d2p_examples directory. */
async function listExampleDatasets() {
	const entries = await fetchJson(IMG_EXAMPLES_CONTENTS_URL);
	
	if (!Array.isArray(entries)) return [];
	return entries
		.filter((entry) => entry?.type === 'dir' && entry.name)
		.map((entry) => entry.name)
		.sort(sortByName);
}

/** Load lightweight manifest summaries for every listed dataset folder. */
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

/** Load the dataset manifest for one dataset and cache the result. */
async function loadDatasetManifest(dataset) {
	if (DATASET_MANIFEST_CACHE.has(dataset)) return DATASET_MANIFEST_CACHE.get(dataset);

	const promise = fetchJson(datasetAssetUrl(dataset, `${dataset}_examples.json`));
	DATASET_MANIFEST_CACHE.set(dataset, promise);
	return promise;
}


/** Open the pair-list workflow for one concrete dataset. */
export async function openDatasetPairList(datasetName, state, controller) {
	state.currentDatasetName = datasetName;
	state.view = 'pairs';

	const datasetLang = csvValueLabel('dataset', datasetName);

	controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
		`<div class="example-loading" ${getTitleAttr('examples.loading.pairs', { datasetName: datasetLang })}>
			${tLabel('examples.loading.pairs', { datasetName: datasetLang })}
		</div>`,
		{
			showBackButton: true,
			backLabel: { key: 'examples.back-to-datasets' },
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
			controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
				`<div class="example-empty-state" ${getTitleAttr('examples.empty.pairs', { datasetName: datasetLang })}>
					${tLabel('examples.empty.pairs', { datasetName: datasetLang })}
				</div>`,
				{
					showBackButton: true,
					backLabel: { key: 'examples.back-to-datasets' },
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
		controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
			`<div class="example-empty-state" ${getTitleAttr('examples.failed.pairs', { datasetName: datasetLang })}>
				${tLabel('examples.failed.pairs', { datasetName: datasetLang })}
			</div>`,
			{
				showBackButton: true,
				backLabel: { key: 'examples.back-to-datasets' },
				onBack: () => {
					renderDatasetChooserView(state, controller);
					controller.focusExamplesViewer();
				}
			}
		);
	}
}

/** Render the dataset chooser. */
export function renderDatasetChooserView(state, controller) {
	state.view = 'datasets';
	const summaries = state.datasetSummaries.filter((item) => item.count > 0);

	controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.choose-dataset' },
		`
			<p class="muted-note mb-4" ${getTitleAttr('examples.dataset.detail')}>
				${tLabel('examples.dataset.detail')}
			</p>
			<div class="dataset-summary-grid">
				${summaries.map((summary) => renderDatasetSummaryCard(summary)).join('')}
			</div>
		`,
		{ showBackButton: false }
	);
}

/** Render one dataset summary card. */
function renderDatasetSummaryCard(summary) {
	return `
		<article class="dataset-summary-card">
			<div>
				<h4 class="title is-5 mb-1" ${getTitleAttr(csvValue('dataset', summary.dataset))}>
					${csvValueLabel('dataset', summary.dataset)}
				</h4>
				<p class="dataset-summary-subtitle mb-0" ${getTitleAttr('examples.dataset.detail.count', { count: summary.count })}>
					${tLabel('examples.dataset.detail.count', { count: summary.count })}
				</p>
			</div>

			<div class="dataset-summary-actions">
				<button class="button is-link is-light" type="button" ${getTitleAttr('examples.dataset.detail.open')} data-open-dataset="${summary.dataset}">
					${tLabel('examples.dataset.detail.open')}
				</button>
			</div>
		</article>
	`;
}


/** Reset all dataset-dependent state before a fresh load. */
function resetDatasetState(state) {
	state.availableDatasets = [];
	state.datasetSummaries = [];
	state.currentDatasetName = null;
	state.currentExamples = [];
	state.view = 'datasets';
}