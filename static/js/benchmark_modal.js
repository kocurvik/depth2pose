const EXAMPLES_BASE_URL = 'https://raw.githubusercontent.com/lbujnak/depth2pose_webdata/main/benchmark/d2p_examples';

const DATASET_MANIFEST_CACHE = new Map();
const DATASET_EXAMPLES_CACHE = new Map();

let modalEls = null;
let modalState = {
	row: null,
	title: '',
	sourceIsMean: false,
	availableDatasets: [],
	datasetSummaries: [],
	currentDatasetName: null,
	currentExamples: [],
	backAction: null
};



/* Initialize the benchmark examples modal and open it whenever a benchmark row is clicked. */
export function initBenchmarkExamplesModal() {
	ensureModal();

	document.addEventListener('benchmark:row-click', (event) => {
		const detail = event.detail || {};
		const { row } = detail;

		if (!row) return;
		openExamplesModal(row, detail);
	});
}

/* Ensure that the benchmark modal exists in the DOM and cache its key elements. */
function ensureModal() {
	if (modalEls) return modalEls;

	const root = document.createElement('div');
	root.id = 'benchmarkExamplesModal';
	root.className = 'benchmark-modal';
	root.hidden = true;
	root.innerHTML = `
		<div class="benchmark-modal-backdrop" data-modal-close></div>
		<div class="benchmark-modal-panel" role="dialog" aria-modal="true" aria-labelledby="benchmarkModalTitle">
			<button class="benchmark-modal-close" type="button" aria-label="Close" data-modal-close>
				<i class="pi pi-times"></i>
			</button>

			<div class="benchmark-modal-inner">
				<div class="benchmark-modal-header">
					<div class="benchmark-modal-header-main">
						<div class="benchmark-modal-heading">
							<h2 class="title is-4 mb-1" id="benchmarkModalTitle">Example viewer</h2>
							<p class="muted-note mb-0" id="benchmarkModalSubtitle"></p>
						</div>

						<button class="benchmark-modal-back" type="button" id="benchmarkModalBack" hidden>
							<span aria-hidden="true">←</span>
							<span>Back</span>
						</button>
					</div>
				</div>

				<div class="benchmark-modal-content" id="benchmarkModalContent"></div>
			</div>
		</div>
	`;

	document.body.appendChild(root);
	root.addEventListener('click', handleModalClick);

	document.addEventListener('keydown', (event) => {
		if (event.key === 'Escape' && !root.hidden) closeModal();
	});

	const back = root.querySelector('#benchmarkModalBack');
	back.addEventListener('click', () => {
		if (typeof modalState.backAction === 'function') modalState.backAction();
	});

	modalEls = {
		root, back,
		title: root.querySelector('#benchmarkModalTitle'),
		subtitle: root.querySelector('#benchmarkModalSubtitle'),
		content: root.querySelector('#benchmarkModalContent')
	};

	return modalEls;
}

/* Open the modal shell with the provided title, subtitle, body, and optional back button state. */
function openModalShell(titleText, subtitleText, bodyHtml, options = {}) {
	const els = ensureModal();
	const { showBackButton = false, backLabel = 'Back', onBack = null } = options;

	document.documentElement.classList.add('modal-open');
	document.body.classList.add('modal-open');
	els.root.hidden = false;
	els.title.textContent = titleText;
	els.subtitle.textContent = subtitleText || '';
	els.content.innerHTML = bodyHtml;

	modalState.backAction = typeof onBack === 'function' ? onBack : null;
	els.back.hidden = !showBackButton;
	els.back.style.display = showBackButton ? 'inline-flex' : 'none';
	els.back.querySelector('span:last-child').textContent = backLabel;
}

/* Handle click actions inside the modal such as close, dataset selection, and pair opening. */
function handleModalClick(event) {
	if (event.target.closest('[data-modal-close]')) {
		closeModal();
		return;
	}

	const datasetButton = event.target.closest('[data-open-dataset]');
	if (datasetButton) {
		const datasetName = datasetButton.getAttribute('data-open-dataset');
		if (datasetName) openDatasetPairList(datasetName);
		return;
	}

	const pairButton = event.target.closest('[data-open-pair]');
	if (pairButton) {
		const pairKey = pairButton.getAttribute('data-open-pair');
		if (pairKey) renderPairDetailView(pairKey);
	}
}

/* Close the modal and clear the currently registered back action. */
function closeModal() {
	if (!modalEls) return;

	modalEls.root.hidden = true;
	modalState.backAction = null;
	document.body.classList.remove('modal-open');
	document.documentElement.classList.remove('modal-open');
}



/* Fetch a JSON file from the given URL and return its parsed contents. */
async function fetchJson(url) {
	const response = await fetch(url);

	if (!response.ok) throw new Error(`Failed to load ${url}: ${response.status}`);
	return response.json();
}

/* Load the dataset manifest for one dataset and cache the result. */
async function loadDatasetManifest(dataset) {
	if (DATASET_MANIFEST_CACHE.has(dataset)) return DATASET_MANIFEST_CACHE.get(dataset);

	const url = `${EXAMPLES_BASE_URL}/${dataset}/${dataset}_examples.json`;
	const promise = fetchJson(url);

	DATASET_MANIFEST_CACHE.set(dataset, promise);
	return promise;
}

/* Load all detailed examples for one dataset and cache the parsed example objects. */
async function loadDetailedExamples(dataset) {
	if (DATASET_EXAMPLES_CACHE.has(dataset)) return DATASET_EXAMPLES_CACHE.get(dataset);

	const promise = (async () => {
		const manifest = await loadDatasetManifest(dataset);
		const entries = Object.entries(manifest);

		const examples = await Promise.all(entries.map(async ([pairKey, baseEntry]) => {
			const resultPath = baseEntry.result_path || `results/${normalizePairKey(pairKey)}.json`;
			const detailsUrl = `${EXAMPLES_BASE_URL}/${dataset}/${resultPath}`;
			const details = await fetchJson(detailsUrl);

			return {
				pairKey, details,
				rgb1Url: `${EXAMPLES_BASE_URL}/${dataset}/${baseEntry.exported_rgb_image1_path}`,
				rgb2Url: `${EXAMPLES_BASE_URL}/${dataset}/${baseEntry.exported_rgb_image2_path}`,
			};
		}));

		return examples;
	})();

	DATASET_EXAMPLES_CACHE.set(dataset, promise);
	return promise;
}

/* Load summary statistics for every available dataset when the clicked row comes from Mean over datasets. */
async function loadMeanDatasetSummaries() {
	const datasets = modalState.availableDatasets || [];

	const summaries = await Promise.all(datasets.map(async (datasetName) => {
		try {
			const examples = await loadDetailedExamples(datasetName);
			const matches = getMatchingExamples(examples, modalState.row.mde, modalState.row.solver);

			if (!matches.length) {
				return { dataset: datasetName, count: 0, meanError: null, bestError: null, worstError: null };
			}

			const errors = matches.map((example) => getPoseError(example.selectedResult)).filter((value) => Number.isFinite(value));

			return {
				dataset: datasetName, count: matches.length,
				meanError: errors.length ? (errors.reduce((sum, value) => sum + value, 0) / errors.length) : null,
				bestError: errors.length ? Math.min(...errors) : null,
				worstError: errors.length ? Math.max(...errors) : null
			};
		}
		catch {
			return { dataset: datasetName, count: 0, meanError: null, bestError: null, worstError: null };
		}
	}));

	return summaries.sort((a, b) => {
		const aErr = Number.isFinite(a.meanError) ? a.meanError : Number.POSITIVE_INFINITY;
		const bErr = Number.isFinite(b.meanError) ? b.meanError : Number.POSITIVE_INFINITY;

		if (aErr !== bErr) return aErr - bErr;
		return String(a.dataset).localeCompare(String(b.dataset));
	});
}



/* Open the examples modal for the clicked benchmark row and branch into dataset or pair browsing. */
async function openExamplesModal(row, detail) {
	modalState = {
		row: { ...row }, title: `${row.mde} · ${row.solver}`,
		sourceIsMean: isMeanDataset(row.dataset) || String(detail.currentDataset || '').toLowerCase() === 'mean',
		availableDatasets: Array.isArray(detail.availableDatasets) ? detail.availableDatasets : [],
		datasetSummaries: [], currentDatasetName: null, currentExamples: [], backAction: null
	};

	if (modalState.sourceIsMean) {
		openModalShell(
			modalState.title, buildSubtitle(row, null),
			`<div class="example-loading">Loading available datasets for this mean row…</div>`
		);

		try {
			const summaries = await loadMeanDatasetSummaries();
			modalState.datasetSummaries = summaries.filter((item) => item.count > 0);

			if (!modalState.datasetSummaries.length) {
				renderInfoState('No example pairs are currently available for this mean row.');
				return;
			}

			renderDatasetChooserView();
		}
		catch (error) {
			console.error(error);
			renderInfoState('Failed to load dataset summaries for this mean row.');
		}

		return;
	}

	await openDatasetPairList(String(row.dataset));
}

/* Open the pair list for one concrete dataset and attach the correct back navigation if needed. */
async function openDatasetPairList(datasetName) {
	modalState.currentDatasetName = datasetName;

	openModalShell(
		modalState.title, buildSubtitle(modalState.row, datasetName),
		`<div class="example-loading">Loading RGB example pairs for ${escapeHtml(datasetName)}…</div>`,
		{
			showBackButton: modalState.sourceIsMean,
			backLabel: 'Back to datasets',
			onBack: () => renderDatasetChooserView()
		}
	);

	try {
		const examples = await loadDetailedExamples(datasetName);
		const matches = getMatchingExamples(examples, modalState.row.mde, modalState.row.solver);

		modalState.currentExamples = matches;

		if (!matches.length) {
			openModalShell(
				modalState.title, buildSubtitle(modalState.row, datasetName),
				`<div class="example-empty-state">No example pairs are currently available for ${escapeHtml(modalState.row.mde)} with solver ${escapeHtml(modalState.row.solver)} on dataset ${escapeHtml(datasetName)}.</div>`,
				{
					showBackButton: modalState.sourceIsMean,
					backLabel: 'Back to datasets',
					onBack: () => renderDatasetChooserView()
				}
			);
			return;
		}

		renderPairListView();
	}
	catch (error) {
		console.error(error);
		openModalShell(
			modalState.title, buildSubtitle(modalState.row, datasetName),
			`<div class="example-empty-state">Failed to load example pairs for dataset ${escapeHtml(datasetName)}.</div>`,
			{
				showBackButton: modalState.sourceIsMean,
				backLabel: 'Back to datasets',
				onBack: () => renderDatasetChooserView()
			}
		);
	}
}



/* Render a simple informational state inside the modal body. */
function renderInfoState(message) {
	openModalShell(
		modalState.title, buildSubtitle(modalState.row, modalState.currentDatasetName),
		`<div class="example-empty-state">${escapeHtml(message)}</div>`
	);
}

/* Render the dataset chooser for Mean over datasets rows. */
function renderDatasetChooserView() {
	const row = modalState.row;
	const summaries = modalState.datasetSummaries;

	openModalShell(
		modalState.title, buildSubtitle(row, null),
		`
			<p class="muted-note mb-4">This row is aggregated over datasets. Choose a concrete dataset to browse RGB pairs first, then open depth details for a selected pair.</p>

			<div class="dataset-summary-grid">
				${summaries.map((summary) => renderDatasetSummaryCard(summary)).join('')}
			</div>
		`,
		{ showBackButton: false }
	);
}

/* Render one dataset summary card in the Mean over datasets chooser. */
function renderDatasetSummaryCard(summary) {
	return `
		<article class="dataset-summary-card">
			<div>
				<h4 class="title is-5 mb-1">${escapeHtml(summary.dataset)}</h4>
				<p class="dataset-summary-subtitle mb-0">${summary.count} RGB pairs available</p>
			</div>

			<div class="dataset-summary-metrics">
				<div class="metric-pill">
					<span class="metric-pill-label">Mean p_err</span>
					<strong>${formatMetric(summary.meanError)}</strong>
				</div>
				<div class="metric-pill">
					<span class="metric-pill-label">Best</span>
					<strong>${formatMetric(summary.bestError)}</strong>
				</div>
				<div class="metric-pill">
					<span class="metric-pill-label">Worst</span>
					<strong>${formatMetric(summary.worstError)}</strong>
				</div>
			</div>

			<div class="dataset-summary-actions">
				<button class="button is-link is-light" type="button" data-open-dataset="${escapeHtml(summary.dataset)}">
					Open pairs
				</button>
			</div>
		</article>
	`;
}

/* Render the list of RGB pairs for the currently selected dataset. */
function renderPairListView() {
	const row = modalState.row;
	const examples = [...modalState.currentExamples].sort((a, b) => getPoseError(b.selectedResult) - getPoseError(a.selectedResult));

	openModalShell(
		modalState.title, buildSubtitle(row, modalState.currentDatasetName),
		`
			<p class="muted-note mb-4">Pairs are ordered by descending <code>p_err</code>, from worst to best. Click a pair to open depth comparison.</p>

			<div class="example-card-grid">
				${examples.map((example) => renderPairCard(example)).join('')}
			</div>
		`,
		{
			showBackButton: modalState.sourceIsMean,
			backLabel: 'Back to datasets',
			onBack: () => renderDatasetChooserView()
		}
	);

	decoratePairCards();
}

/* Render one RGB pair card with its statistics and correspondence overlay. */
function renderPairCard(example) {
	const selected = example.selectedResult;
	const pErr = selected?.p_err;
	const baselineErr = example.details?.baseline_p_err;
	const unusedCount = Array.isArray(selected?.unused_kps) ? selected.unused_kps.length : 0;
	const totalCount = Array.isArray(example.details?.kp1) ? example.details.kp1.length : 0;
	const inlierCount = countInliers(selected?.inliers, totalCount);
	const outlierCount = Math.max(totalCount - inlierCount - unusedCount, 0);

	const safeExample = encodeURIComponent(JSON.stringify({
		kp1: example.details?.kp1 || [], kp2: example.details?.kp2 || [], inliers: selected?.inliers || [], unusedKps: selected?.unused_kps || []
	}));

	return `
		<article class="example-card">
			<div class="example-card-header">
				<div>
					<h4 class="title is-6 mb-1">${escapeHtml(shortPairName(example.pairKey))}</h4>
					<p class="example-meta mb-0">p_err ${formatMetric(pErr)} · baseline ${formatMetric(baselineErr)}</p>
				</div>
			</div>

			<div class="example-stat-pills">
				<span class="example-stat-pill is-inlier">${inlierCount} inliers</span>
				<span class="example-stat-pill is-outlier">${outlierCount} outliers</span>
				<span class="example-stat-pill is-unused">${unusedCount} unused</span>
			</div>

			<div class="example-visual-pair" data-example='${safeExample}'>
				<div class="example-image-panel">
					<img class="example-rgb-image" src="${example.rgb1Url}" alt="RGB image 1 for ${escapeHtml(example.pairKey)}">
					<div class="example-image-label">Image 1</div>
				</div>

				<div class="example-image-panel">
					<img class="example-rgb-image" src="${example.rgb2Url}" alt="RGB image 2 for ${escapeHtml(example.pairKey)}">
					<div class="example-image-label">Image 2</div>
				</div>

				<svg class="example-correspondence-overlay" aria-hidden="true"></svg>
			</div>

			<div class="example-card-actions">
				<button class="button is-link is-light is-small" type="button" data-open-pair="${escapeHtml(example.pairKey)}">
					Show depth details
				</button>
			</div>
		</article>
	`;
}

/* Render the detail view for one selected pair together with ordered depth results. */
function renderPairDetailView(pairKey) {
	const example = modalState.currentExamples.find((item) => item.pairKey === pairKey);
	if (!example) return;

	const preferredResults = getComparableResults(example, modalState.row.solver);
	const overviewResult = example.selectedResult || preferredResults[0]?.result || null;

	const totalCount = Array.isArray(example.details?.kp1) ? example.details.kp1.length : 0;
	const unusedCount = Array.isArray(overviewResult?.unused_kps) ? overviewResult.unused_kps.length : 0;
	const inlierCount = countInliers(overviewResult?.inliers, totalCount);
	const outlierCount = Math.max(totalCount - inlierCount - unusedCount, 0);

	const safeOverview = encodeURIComponent(JSON.stringify({
		kp1: example.details?.kp1 || [], kp2: example.details?.kp2 || [], inliers: overviewResult?.inliers || [], unusedKps: overviewResult?.unused_kps || []
	}));

	openModalShell(
		modalState.title, buildSubtitle(modalState.row, modalState.currentDatasetName),
		`
			<div class="pair-detail-header">
				<div>
					<h3 class="title is-5 mb-1">${escapeHtml(shortPairName(example.pairKey))}</h3>
					<p class="muted-note mb-0">Depth results are ordered by ascending <code>p_err</code>, from best to worst for this pair.</p>
				</div>
			</div>

			<div class="example-stat-pills example-stat-pills-overview">
				<span class="example-stat-pill is-inlier">${inlierCount} inliers</span>
				<span class="example-stat-pill is-outlier">${outlierCount} outliers</span>
				<span class="example-stat-pill is-unused">${unusedCount} unused</span>
			</div>

			<div class="pair-detail-overview">
				<div class="example-visual-pair example-visual-pair-large" data-example='${safeOverview}'>
					<div class="example-image-panel">
						<img class="example-rgb-image" src="${example.rgb1Url}" alt="RGB image 1 for ${escapeHtml(example.pairKey)}">
						<div class="example-image-label">Image 1</div>
					</div>

					<div class="example-image-panel">
						<img class="example-rgb-image" src="${example.rgb2Url}" alt="RGB image 2 for ${escapeHtml(example.pairKey)}">
						<div class="example-image-label">Image 2</div>
					</div>

					<svg class="example-correspondence-overlay" aria-hidden="true"></svg>
				</div>
			</div>

			<div class="depth-result-stack">
				${preferredResults.map((item, index) => renderDepthResultCard(item, index, modalState.row.mde, modalState.currentDatasetName, example)).join('')}
			</div>
		`,
		{
			showBackButton: true,
			backLabel: 'Back to pairs',
			onBack: () => renderPairListView()
		}
	);

	decoratePairCards();
}

/* Render one depth result card for one MDE on the currently selected pair. */
function renderDepthResultCard(item, index, currentMde, dataset, example) {
	const { mdeName, result } = item;
	const depth1 = result?.exported_depth_image1_path ? `${EXAMPLES_BASE_URL}/${dataset}/${result.exported_depth_image1_path}` : '';
	const depth2 = result?.exported_depth_image2_path ? `${EXAMPLES_BASE_URL}/${dataset}/${result.exported_depth_image2_path}` : '';

	const totalCount = Array.isArray(example?.details?.kp1) ? example.details.kp1.length : 0;
	const unusedCount = Array.isArray(result?.unused_kps) ? result.unused_kps.length : 0;
	const inlierCount = countInliers(result?.inliers, totalCount);
	const outlierCount = Math.max(totalCount - inlierCount - unusedCount, 0);

	const isCurrent = String(mdeName) === String(currentMde);

	return `
		<article class="depth-result-card ${isCurrent ? 'is-current' : ''}">
			<div class="depth-result-meta">
				<div class="depth-result-rank">#${index + 1}</div>
				<div class="depth-result-text">
					<h4 class="title is-6 mb-1">${escapeHtml(mdeName)}</h4>
					<p class="depth-result-line mb-0">
						<span><strong>solver:</strong> ${escapeHtml(String(result?.solver ?? '—'))}</span>
						<span><strong>p_err:</strong> ${formatMetric(result?.p_err)}</span>
					</p>
				</div>
			</div>

			<div class="example-stat-pills example-stat-pills-overview">
				<span class="example-stat-pill is-inlier">${inlierCount} inliers</span>
				<span class="example-stat-pill is-outlier">${outlierCount} outliers</span>
				<span class="example-stat-pill is-unused">${unusedCount} unused</span>
			</div>

			<div class="example-depth-grid">
				${depth1 ? `<figure class="example-depth-panel"><img src="${depth1}" alt="Depth map 1 for ${escapeHtml(mdeName)}"><figcaption>Depth 1 · ${escapeHtml(mdeName)}</figcaption></figure>` : '<div class="example-depth-panel example-depth-empty">Depth 1 not available</div>'}
				${depth2 ? `<figure class="example-depth-panel"><img src="${depth2}" alt="Depth map 2 for ${escapeHtml(mdeName)}"><figcaption>Depth 2 · ${escapeHtml(mdeName)}</figcaption></figure>` : '<div class="example-depth-panel example-depth-empty">Depth 2 not available</div>'}
			</div>
		</article>
	`;
}



/* Decorate every currently visible RGB pair panel with correspondence overlays after images load. */
function decoratePairCards() {
	const cards = document.querySelectorAll('.example-visual-pair[data-example]');

	cards.forEach((card) => {
		const payload = card.getAttribute('data-example');
		if (!payload) return;

		let data;
		try {
			data = JSON.parse(decodeURIComponent(payload));
		}
		catch {
			return;
		}

		const imgs = card.querySelectorAll('.example-rgb-image');
		const svg = card.querySelector('.example-correspondence-overlay');
		if (imgs.length < 2 || !svg) return;

		const draw = () => drawCorrespondences(card, imgs[0], imgs[1], svg, data);

		let remaining = 0;
		imgs.forEach((img) => {
			if (!img.complete) {
				remaining += 1;
				img.addEventListener('load', () => {
					remaining -= 1;
					if (remaining <= 0) draw();
				}, { once: true });
			}
		});

		if (remaining === 0) draw();
	});
}

/* Draw inlier, outlier, and unused correspondence overlays for one RGB pair card. */
function drawCorrespondences(container, img1, img2, svg, data) {
	const containerRect = container.getBoundingClientRect();
	const rect1 = img1.getBoundingClientRect();
	const rect2 = img2.getBoundingClientRect();

	if (!rect1.width || !rect2.width || !img1.naturalWidth || !img2.naturalWidth) return;

	const total = Math.min(data.kp1.length, data.kp2.length);
	const unusedSet = new Set(Array.isArray(data.unusedKps) ? data.unusedKps.map(Number) : []);
	const inlierSet = buildInlierSet(total, data.inliers);

	const inlierIdx = [];
	const outlierIdx = [];
	const unusedIdx = [];

	for (let i = 0; i < total; i += 1) {
		if (unusedSet.has(i)) unusedIdx.push(i);
		else if (inlierSet.has(i)) inlierIdx.push(i);
		else outlierIdx.push(i);
	}

	const selectedInliers = evenlySample(inlierIdx, 60);
	const selectedOutliers = evenlySample(outlierIdx, 80);
	const selectedUnused = evenlySample(unusedIdx, 24);

	svg.setAttribute('viewBox', `0 0 ${Math.round(containerRect.width)} ${Math.round(containerRect.height)}`);
	svg.innerHTML = '';

	const appendLine = (x1, y1, x2, y2, kind) => {
		const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
		line.setAttribute('x1', x1);
		line.setAttribute('y1', y1);
		line.setAttribute('x2', x2);
		line.setAttribute('y2', y2);
		line.setAttribute('class', `corr-line ${kind}`);
		svg.appendChild(line);
	};

	const appendCircle = (x, y, kind) => {
		const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
		circle.setAttribute('cx', x);
		circle.setAttribute('cy', y);
		circle.setAttribute('r', kind === 'unused' ? 1.8 : 2.1);
		circle.setAttribute('class', `corr-point ${kind}`);
		svg.appendChild(circle);
	};

	selectedInliers.forEach((index) => {
		const [x1, y1] = projectPoint(data.kp1[index], rect1, containerRect, img1);
		const [x2, y2] = projectPoint(data.kp2[index], rect2, containerRect, img2);

		appendLine(x1, y1, x2, y2, 'inlier');
		appendCircle(x1, y1, 'inlier');
		appendCircle(x2, y2, 'inlier');
	});

	selectedOutliers.forEach((index) => {
		const [x1, y1] = projectPoint(data.kp1[index], rect1, containerRect, img1);
		const [x2, y2] = projectPoint(data.kp2[index], rect2, containerRect, img2);

		appendLine(x1, y1, x2, y2, 'outlier');
		appendCircle(x1, y1, 'outlier');
		appendCircle(x2, y2, 'outlier');
	});

	selectedUnused.forEach((index) => {
		const [x1, y1] = projectPoint(data.kp1[index], rect1, containerRect, img1);
		const [x2, y2] = projectPoint(data.kp2[index], rect2, containerRect, img2);

		appendCircle(x1, y1, 'unused');
		appendCircle(x2, y2, 'unused');
	});
}



/* Return only those examples that contain the requested exact MDE and exact solver. */
function getMatchingExamples(examples, mdeName, solverName) {
	return examples
		.filter((example) => {
			const result = example.details?.results?.[mdeName];
			return result && String(result.solver) === String(solverName);
		})
		.map((example) => ({ ...example, selectedResult: example.details.results[mdeName] }));
}

/* Return comparable results for one pair, preferring rows with the same solver as the clicked benchmark row. */
function getComparableResults(example, solverName) {
	const allResults = Object.entries(example.details?.results || {})
		.map(([mdeName, result]) => ({ mdeName, result }))
		.filter(({ result }) => result);

	const sameSolver = allResults.filter(({ result }) => String(result.solver) === String(solverName));
	const chosen = sameSolver.length ? sameSolver : allResults;

	return chosen.sort((a, b) => {
		const aErr = Number(a.result?.p_err ?? Number.POSITIVE_INFINITY);
		const bErr = Number(b.result?.p_err ?? Number.POSITIVE_INFINITY);
		return aErr - bErr;
	});
}

/* Build a subtitle string for the modal based on dataset scope and iteration count. */
function buildSubtitle(row, datasetName) {
	const itersText = row.iters !== undefined ? ` · Iters: ${row.iters}` : '';

	if (datasetName) return `Dataset: ${datasetName}${itersText}`;
	if (isMeanDataset(row.dataset)) return `Mean over datasets${itersText}`;
	return row.dataset ? `Dataset: ${row.dataset}${itersText}` : '';
}

/* Check whether the provided dataset value corresponds to Mean over datasets. */
function isMeanDataset(value) {
	return String(value ?? '').toLowerCase() === 'mean';
}

/* Normalize inlier representation into a set of active keypoint indices. */
function buildInlierSet(total, inliers) {
	if (!Array.isArray(inliers)) return new Set();

	const looksLikeMask = inliers.length === total && inliers.every((value) => (
		typeof value === 'boolean' || value === 0 || value === 1
	));

	if (looksLikeMask) {
		const out = new Set();

		inliers.forEach((value, index) => {
			if (Boolean(value)) out.add(index);
		});
		return out;
	}

	return new Set(inliers.map((value) => Number(value)).filter((value) => Number.isInteger(value) && value >= 0));
}

/* Count how many inliers are present after normalizing mask or index-list formats. */
function countInliers(inliers, total) {
	return buildInlierSet(total, inliers).size;
}

/* Project one keypoint from image coordinates into the overlay coordinate system. */
function projectPoint(point, imageRect, containerRect, imageEl) {
	const [xRaw, yRaw] = point || [0, 0];
	const x = imageRect.left - containerRect.left + (Number(xRaw) / imageEl.naturalWidth) * imageRect.width;
	const y = imageRect.top - containerRect.top + (Number(yRaw) / imageEl.naturalHeight) * imageRect.height;
	return [x.toFixed(2), y.toFixed(2)];
}

/* Evenly sample a long list of indices to keep overlays readable. */
function evenlySample(items, maxCount) {
	if (items.length <= maxCount) return items;

	const out = [];
	const step = items.length / maxCount;

	for (let i = 0; i < maxCount; i += 1) {
		out.push(items[Math.floor(i * step)]);
	}

	return out;
}

/* Extract a numeric pose error from a result-like object and return Infinity when missing. */
function getPoseError(resultLike) {
	return Number(resultLike?.p_err ?? resultLike?.best_mde_p_err ?? Number.POSITIVE_INFINITY);
}

/* Format a numeric metric for compact display in the UI. */
function formatMetric(value) {
	if (!Number.isFinite(value)) return '—';
	return Number(value).toFixed(3).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

/* Normalize a pair key so it can be reused as a stable file name fragment. */
function normalizePairKey(pairKey) {
	return String(pairKey ?? '').replaceAll('\\', '_').replaceAll("'", '_');
}

/* Convert a raw pair key into a shorter human-readable display name. */
function shortPairName(pairKey) {
	return String(pairKey || '').replaceAll('\\', ' / ').replaceAll("'", ' / ');
}

/* Escape HTML-sensitive characters for safe text insertion into the modal markup. */
function escapeHtml(value) {
	return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}