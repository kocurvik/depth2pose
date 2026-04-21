let DATASETS = [];

const MODE_CONFIG = {
	calibrated: {
		label: 'With calibration',
		solvers: new Set(['calib', 'calib_shift', 'baseline_calib', 'baseline'])
	},
	uncalibrated: {
		label: 'Without calibration',
		solvers: new Set(['sf', 'sf_shift', 'vf', 'vf_shift', 'mdecalib', 'mdecalib_shift', 'baseline_vf'])
	}
};

const state = {
	datasets: {},
	currentDatasetId: '',
	search: '',
	scoreMetric: 'pose_mAA_10',
	mode: 'calibrated',
	bestBackboneOnly: false,
	bestSolverOnly: false,
	hideGtOnly: true
};

const els = {
	datasetSelect: document.getElementById('datasetSelect'),
	scoreMetricSelect: document.getElementById('scoreMetricSelect'),
	searchInput: document.getElementById('searchInput'),
	bestBackboneOnly: document.getElementById('bestBackboneOnly'),
	bestSolverOnly: document.getElementById('bestSolverOnly'),
	hideGtOnly: document.getElementById('hideGtOnly'),
	resultsBody: document.getElementById('resultsBody'),
	summaryGrid: document.getElementById('summaryGrid'),
	emptyState: document.getElementById('emptyState')
};


// Utility functions
function fmt(value, digits = 2) {
	if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
	return Number(value).toFixed(digits);
}


// Dataset loading and preprocessing
async function loadDatasetList() {
	const owner_repo = 'lbujnak/depth2pose_webdata';
	const branch = 'main';
	const path = 'benchmark';

	const url = `https://api.github.com/repos/${owner_repo}/contents/${path}?ref=${branch}`;

	const res = await fetch(url, { headers: { 'Accept': 'application/vnd.github+json' } });
	if (!res.ok) throw new Error(`Failed to load dataset list: ${res.status}`);

	const items = await res.json();
	
	return items
		.filter(item => item.type === 'file' && item.name.endsWith('.json'))
		.map(item => ({
			id: item.name.replace(/\.json$/, ''),
			label: item.name.replace(/\.json$/, ''),
			file: item.download_url
		}))
		.sort((a, b) => a.label.localeCompare(b.label));
}

function populateDatasetSelect() {
	const select = document.getElementById('datasetSelect');

	select.innerHTML = '';
	for (const ds of DATASETS) {
		const option = document.createElement('option');
		option.value = ds.id;
		option.textContent = ds.label;
		select.appendChild(option);
	}
}


async function loadDatasets() {
	const results = await Promise.all(DATASETS.map(async (dataset) => {
		const response = await fetch(dataset.file);
		if (!response.ok) throw new Error(`Failed to load ${dataset.file}`);

		const json = await response.json();
		return [dataset.id, flattenDataset(json)];
	}));

	state.datasets = Object.fromEntries(results);
}

function flattenDataset(raw) {
	const rows = [];
	Object.entries(raw).forEach(([modelName, solverMap]) => {
		const parsed = parseModelName(modelName);
		Object.entries(solverMap).forEach(([solver, metrics]) => {
			rows.push({
				modelName, family: parsed.family,
				backbone: parsed.backbone, solver,
				solverFamily: solverFamily(solver),
				pose_mAA_10: metrics.pose_mAA_10,
				pose_mAA_5: metrics.pose_mAA_5,
				pose_mAA_3: metrics.pose_mAA_3,
				median_pose_err: metrics.median_pose_err,
				mean_runtime: metrics.mean_runtime,
				mean_mde_runtime: metrics.mean_mde_runtime,
				mean_inliers: metrics.mean_inliers,
			});
		});
	});
	return rows;
}

function parseModelName(modelName) {
	const dashIndex = modelName.indexOf('-');
	if (dashIndex === -1) return { family: modelName, backbone: '—' };
	return { family: modelName.slice(0, dashIndex), backbone: modelName.slice(dashIndex + 1) };
}

function solverFamily(solver) {
	if (solver.endsWith('_shift')) return solver.replace(/_shift$/, '');
	return solver;
}


// Table rendering
function renderTable() {
	const rows = getProcessedRows();
	renderSummary(rows);

	if (!rows.length) {
		els.resultsBody.innerHTML = '';
		els.emptyState.hidden = false;
		els.emptyState.textContent = 'No rows match the current filters.';
		return;
	}

	els.emptyState.hidden = true;
	els.resultsBody.innerHTML = rows.map((row, index) => `
		<tr>
			<td><strong>${index + 1}</strong></td>
			<td>${row.family}</td>
			<td>${row.backbone}</td>
			<td><span class="tag is-soft">${row.modelName}</span></td>
			<td><span class="tag is-soft">${row.solver}</span></td>
			<td ${state.scoreMetric === 'pose_mAA_10' ? 'class="metric-strong"' : ''}>${fmt(row.pose_mAA_10)}</td>
			<td ${state.scoreMetric === 'pose_mAA_5' ? 'class="metric-strong"' : ''}>${fmt(row.pose_mAA_5)}</td>
			<td ${state.scoreMetric === 'pose_mAA_3' ? 'class="metric-strong"' : ''}>${fmt(row.pose_mAA_3)}</td>
			<td>${fmt(row.median_pose_err)}</td>
			<td>${fmt(row.mean_runtime)}</td>
			<td>${fmt(row.mean_mde_runtime)}</td>
			<td>${fmt(row.mean_inliers, 3)}</td>
		</tr>
	`).join('');
}

// Table rendering - get filtered/sorted rows
function getProcessedRows() {
	let rows = [...(state.datasets[state.currentDatasetId] || [])];
	rows = applyCaseLogic(rows);
	rows = applySearch(rows);

	if (state.hideGtOnly) rows = rows.filter((row) => row.modelName !== 'gt');
	if (state.bestSolverOnly) rows = pickBest(rows, (row) => `${row.modelName}__${row.solverFamily}`, state.scoreMetric);
	if (state.bestBackboneOnly) rows = pickBest(rows, (row) => `${row.family}__${row.solverFamily}`, state.scoreMetric);
	return sortRows(rows);
}

function applyCaseLogic(rows) {
	const modeCfg = MODE_CONFIG[state.mode];
	return rows.filter((row) => {
		if (!modeCfg.solvers.has(row.solver)) return false;
		if (state.mode === 'uncalibrated' && row.modelName.includes('Calib')) return false;
		return true;
	});
}

function applySearch(rows) {
	const q = state.search.trim().toLowerCase();
	if (!q) return rows;
	return rows.filter((row) => [row.family, row.backbone, row.modelName, row.solver].join(' ').toLowerCase().includes(q));
}

function pickBest(rows, keyFn, scoreKey) {
	const best = new Map();
	for (const row of rows) {
		const key = keyFn(row), prev = best.get(key);
		if (!prev || betterRow(row, prev, scoreKey)) best.set(key, row);
	}
	return [...best.values()];
}

function betterRow(candidate, current, scoreKey) {
	const candidateScore = candidate[scoreKey] ?? -Infinity;
	const currentScore = current[scoreKey] ?? -Infinity;
	if (candidateScore !== currentScore) return candidateScore > currentScore;

	const tieBreakers = [
		(candidate.pose_mAA_5 ?? -Infinity) - (current.pose_mAA_5 ?? -Infinity),
		(candidate.pose_mAA_3 ?? -Infinity) - (current.pose_mAA_3 ?? -Infinity),
		(current.median_pose_err ?? Infinity) - (candidate.median_pose_err ?? Infinity)
	];
	return tieBreakers.find(v => v !== 0) > 0;
}

function sortRows(rows) {
	return rows.sort((a, b) => {
		const primary = (b[state.scoreMetric] ?? -Infinity) - (a[state.scoreMetric] ?? -Infinity);
		if (primary !== 0) return primary;

		const second = (b.pose_mAA_5 ?? -Infinity) - (a.pose_mAA_5 ?? -Infinity);
		if (second !== 0) return second;

		return (a.median_pose_err ?? Infinity) - (b.median_pose_err ?? Infinity);
	});
}


// Render summary statistics
function renderSummary(rows) {
	const families = new Set(rows.map(r => r.family)).size;
	const backbones = new Set(rows.map(r => `${r.family}__${r.backbone}`)).size;
	const leader = rows[0];
	const stats = [
		['Visible rows', rows.length],
		['Estimator families', families],
		['Visible backbones', backbones],
		[state.scoreMetric + ' leader', leader ? `${leader.modelName} / ${leader.solver}` : '—']
	];

	els.summaryGrid.innerHTML = stats.map(([label, value]) => `
		<div class="summary-box">
			<div class="label-title">${label}</div>
			<div class="value">${value}</div>
		</div>
	`).join('');
}


// UI binding to state and event listeners
function bindControls() {
	els.datasetSelect.value = state.currentDatasetId;
	els.scoreMetricSelect.value = state.scoreMetric;
	els.hideGtOnly.checked = state.hideGtOnly;
	els.bestBackboneOnly.checked = state.bestBackboneOnly;
	els.bestSolverOnly.checked = state.bestSolverOnly;
	document.querySelector(`input[name="mode"][value="${state.mode}"]`).checked = true;

	els.datasetSelect.addEventListener('change', (event) => {
		state.currentDatasetId = event.target.value;
		renderTable();
	});

	els.scoreMetricSelect.addEventListener('change', (event) => {
		state.scoreMetric = event.target.value;
		renderTable();
	});

	els.searchInput.addEventListener('input', (event) => {
		state.search = event.target.value;
		renderTable();
	});

	els.bestBackboneOnly.addEventListener('change', (event) => {
		state.bestBackboneOnly = event.target.checked;
		renderTable();
	});

	els.bestSolverOnly.addEventListener('change', (event) => {
		state.bestSolverOnly = event.target.checked;
		renderTable();
	});

	els.hideGtOnly.addEventListener('change', (event) => {
		state.hideGtOnly = event.target.checked;
		renderTable();
	});

	document.querySelectorAll('input[name="mode"]').forEach((input) => {
		input.addEventListener('change', (event) => {
			state.mode = event.target.value;
			renderTable();
		});
	});
}


// Initialize the viewer
async function init() {
	bindControls();

	try {
		DATASETS = await loadDatasetList();
		state.currentDatasetId = DATASETS[0]?.id ?? '';
		populateDatasetSelect();

		await loadDatasets();
		renderTable();
	}
	catch (error) {
		console.error(error);
		els.emptyState.hidden = false;
		els.emptyState.textContent = 'Failed to load JSON results.';
	}
}

init();