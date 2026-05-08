import { attachCardToggle, isGtMde, isCalibMde, getMdeFamilyKey, isRoSolver, isAISolver, baseSolverName, normalizeForSearch, sortByName, isNumericValue } from "./global.js";
import { STANDARD_TABLE_SOURCES, GROUP_TABLE_SOURCES } from "./api-config.js";
import { tLabel, tDesc, csvValueLabel, csvValueDesc, setOptionTitle, getTitleAttr } from "./dictionary/index.js";
import { t } from "./dictionary/dict.js";

const BENCHMARK_CONFIG = {
	standard: {
		sources: STANDARD_TABLE_SOURCES,
	},
	group: {
		sources: GROUP_TABLE_SOURCES,
	}
};

const MODE_CONFIG = {
	calibrated: {
		solvers: new Set(['calib', 'calib_shift', 'baseline_calib'])
	},
	uncalibrated: {
		solvers: new Set(['sf', 'sf_shift','vf', 'vf_shift','mdecalib', 'mdecalib_shift','baseline_sf', 'baseline_vf'])
	}
};

const state = {
	benchmarkMode: 'group',
	currentSource: Object.keys(GROUP_TABLE_SOURCES)[0],
	rawRows: [],
	columns: [],
	numericColumns: new Set(),
	itersOptions: [],
	datasets: [],
	groups: [],
	currentDataset: 'mean',
	currentIters: '1000',
	mode: 'calibrated',
	roVariant: 'non_ro',
	depthType: 'scale',
	hideGtOnly: true,
	bestMdeOnly: false,
	sortKey: 'pose_mAA_10',
	sortDir: 'desc',
	search: '', page: 1, pageSize: 25, visibleRows: []
};

const els = {
	benchmarkModeSelect: document.getElementById('benchmarkModeSelect'),
	sourceSelect: document.getElementById('sourceSelect'),
	datasetLabel: document.getElementById('datasetLabel'),
	datasetSelect: document.getElementById('datasetSelect'),
	itersSelect: document.getElementById('itersSelect'),
	evaluationCaseSelect: document.getElementById('evaluationCaseSelect'),
	solverVariantSelect: document.getElementById('solverVariantSelect'),
	depthTypeSelect: document.getElementById('depthTypeSelect'),
	searchInput: document.getElementById('searchInput'),
	pageSizeSelect: document.getElementById('pageSizeSelect'),
	hideGtOnly: document.getElementById('hideGtOnly'),
	bestMdeOnly: document.getElementById('bestMdeOnly'),
	resultsHead: document.getElementById('resultsHead'),
	resultsBody: document.getElementById('resultsBody'),
	summaryGrid: document.getElementById('summaryGrid'),
	emptyState: document.getElementById('emptyState'),
	paginationBar: document.getElementById('paginationBar'),
	paginationInfo: document.getElementById('paginationInfo'),
	prevPageBtn: document.getElementById('prevPageBtn'),
	nextPageBtn: document.getElementById('nextPageBtn')
};


/** Initialize the benchmark viewer */
export async function initResultsTable() {
	attachCardToggle('controlsCard', 'controlsToggle', 'controlsContent', { key: 'controls.toggle.show' }, { key: 'controls.toggle.hide' });
	attachCardToggle('tableCard', 'tableToggle', 'tableContent', { key: 'table.toggle.show' }, { key: 'table.toggle.hide' });
	attachCardToggle('resultsCard', 'resultsToggle', 'resultsContent', { key: 'results.toggle.show' }, { key: 'results.toggle.hide' });

	try {
		await loadData();
		populateControls();
		bindControls();
		render();
	}
	catch (error) {
		console.error(error);
		setTableErrorState({ key: 'table.loadFail' });
	}
}


/** Load CSV data from the specified URL, parse it, and populate the state with columns, rows, and options for controls. */
async function loadData() {
	let csvText;

	try {
		const sources = BENCHMARK_CONFIG[state.benchmarkMode].sources;
		const url = sources[state.currentSource]?.url;
		if (!url) throw new Error(`Unknown source: ${state.currentSource}`);
		
		const response = await fetch(url);
		if (!response.ok) throw new Error(`Failed to load CSV from ${url}: ${response.status}`);
		
		csvText = await response.text();
	}
	catch (error) {
		throw new Error('Failed to load CSV.');
	}

	const { columns, rows } = parseCsv(csvText);

	state.columns = columns;
	state.rawRows = rows;
	state.numericColumns = new Set(columns.filter((column) => rows.every((row) => typeof row[column] === 'number' || row[column] === '')));
	state.datasets = [...new Set(rows.map((row) => String(row.dataset)).filter(Boolean))].sort(sortByName);
	state.groups = columns.includes('group') ? [...new Set(rows.map((row) => String(row.group)).filter(Boolean))].sort(sortByName) : [];
	state.itersOptions = [...new Set(rows.map((row) => String(row.iters)))].sort((a, b) => Number(a) - Number(b));
}

/** Parses a CSV string into columns and rows. */
function parseCsv(text) {
	const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n').filter(line => line.trim().length);
	if (!lines.length) return { columns: [], rows: [] };

	const columns = parseCsvLine(lines[0]).map(col => col.trim());
	const rows = lines.slice(1).map((line) => {
		const values = parseCsvLine(line);
		const row = {};

		columns.forEach((column, index) => {
			const raw = (values[index] ?? '').trim();
			row[column] = isNumericValue(raw) ? Number(raw) : raw;
		});
		return row;
	});

	return { columns, rows };
}

/** Parses a single line of CSV, handling quoted values and commas. */
function parseCsvLine(line) {
	const out = [];
	let current = '';
	let inQuotes = false;

	for (let i = 0; i < line.length; i += 1) {
		const char = line[i];
		const next = line[i + 1];

		if (char === '"') {
			if (inQuotes && next === '"') {
				current += '"';
				i += 1;
			}
			else inQuotes = !inQuotes;
		}
		else if (char === ',' && !inQuotes) {
			out.push(current);
			current = '';
		}
		else current += char;
	}

	out.push(current);
	return out;
}


/** Populate the dataset and iters dropdowns based on the loaded data, and set initial values for all controls. */
function populateControls() {
	const bm = state.benchmarkMode;

	// Dataset label
	if (els.datasetLabel) {
		const key = 'controls.label.dataset.' + bm;
		els.datasetLabel.innerHTML = tLabel(key);
		setOptionTitle(els.datasetLabel, tDesc(key));
	}

	// Benchmark mode select
	if (els.benchmarkModeSelect) {
		els.benchmarkModeSelect.innerHTML = '';
		for (const mode of Object.keys(BENCHMARK_CONFIG)) {
			const key = `controls.select.benchmark.${mode}`;
			const option = document.createElement('option');
			
			option.value = mode;
			option.textContent = tLabel(key);
			setOptionTitle(option, tDesc(key));
			els.benchmarkModeSelect.appendChild(option);
		}
		els.benchmarkModeSelect.value = state.benchmarkMode;
	}

	populateSourceSelect();
	populateDatasetSelect();

	// Iters select
	const allKey = 'controls.select.iters.all';
	const allOpt = document.createElement('option');
	allOpt.value = 'all';
	allOpt.textContent = tLabel(allKey);
	setOptionTitle(allOpt, tDesc(allKey));

	els.itersSelect.innerHTML = '';
	els.itersSelect.appendChild(allOpt);
	
	state.itersOptions.forEach((iters) => {
		const option = document.createElement('option');
		option.value = iters;
		option.textContent = iters;
		els.itersSelect.appendChild(option);
	});

	// Evaluation case select
	els.evaluationCaseSelect.innerHTML = '';
	for (const val of ['all', 'calibrated', 'uncalibrated']) {
		const key = `controls.select.evaluationCase.${val}`;
		const option = document.createElement('option');
		
		option.value = val;
		option.textContent = tLabel(key);
		setOptionTitle(option, tDesc(key));
		els.evaluationCaseSelect.appendChild(option);
	}

	// Solver variant select
	els.solverVariantSelect.innerHTML = '';
	for (const val of ['all', 'non_ro', 'ro']) {
		const key = `controls.select.solverVariant.${val}`;
		const option = document.createElement('option');
		
		option.value = val;
		option.textContent = tLabel(key);
		setOptionTitle(option, tDesc(key));
		els.solverVariantSelect.appendChild(option);
	}

	// Depth type select
	els.depthTypeSelect.innerHTML = '';
	for (const val of ['all', 'scale', 'affine']) {
		const key = `controls.select.depthType.${val}`;
		const option = document.createElement('option');
		
		option.value = val;
		option.textContent = tLabel(key);
		setOptionTitle(option, tDesc(key));
		els.depthTypeSelect.appendChild(option);
	}

	// Set current values
	els.datasetSelect.value = state.currentDataset;
	els.itersSelect.value = state.currentIters;
	els.evaluationCaseSelect.value = state.mode;
	els.solverVariantSelect.value = state.roVariant;
	els.pageSizeSelect.value = String(state.pageSize);
	els.hideGtOnly.checked = state.hideGtOnly;
	els.bestMdeOnly.checked = state.bestMdeOnly;
}

/** Populate the source dropdown based on the available sources for the current benchmark mode. */
function populateSourceSelect() {
	if (!els.sourceSelect) return;

	const sources = BENCHMARK_CONFIG[state.benchmarkMode].sources;
	const sourceEntries = Object.entries(sources);
	els.sourceSelect.innerHTML = '';

	sourceEntries.forEach(([value, config]) => {
		const option = document.createElement('option');
		
		option.value = value;
		option.textContent = tLabel(config.key);
		setOptionTitle(option, tDesc(config.key));
		els.sourceSelect.appendChild(option);
	});

	els.sourceSelect.value = state.currentSource;
}

/** Populate the dataset/group/scene dropdown for the selected benchmark source. */
function populateDatasetSelect() {
	const bm = state.benchmarkMode;
	els.datasetSelect.innerHTML = '';

	// Mean option
	const meanKey = `controls.select.dataset.mean.${bm}`;
	appendDatasetOption('mean', tLabel(meanKey), els.datasetSelect, tDesc(meanKey));

	if (bm !== 'group') {
		state.datasets.forEach((ds) => {
			appendDatasetOption(ds, csvValueLabel('dataset', ds), els.datasetSelect, csvValueDesc('dataset', ds));
		});
		return;
	}

	const prefix = tLabel('controls.select.dataset.meanOverGroupPrefix');
	appendDatasetOptGroup(
		`${prefix}…`,
		state.groups.map((group) => ({
			value: `group:${group}`,
			label: `${prefix} ${csvValueLabel('group', group)}`,
			description: csvValueDesc('group', group)
		}))
	);

	appendDatasetOptGroup(
		'Scenes',
		state.datasets.map((ds) => ({
			value: `scene:${ds}`,
			label: csvValueLabel('scene', ds),
			description: csvValueDesc('scene', ds)
		}))
	);
}

/** Append one option to the dataset selector. */
function appendDatasetOption(value, label, parent = els.datasetSelect, description = null) {
	const option = document.createElement('option');
	option.value = value;
	option.textContent = label;
	
	setOptionTitle(option, description);
	parent.appendChild(option);
}

/** Append a labelled option group when it has at least one option. */
function appendDatasetOptGroup(label, options) {
	if (!options.length) return;

	const group = document.createElement('optgroup');
	group.label = label;

	options.forEach(({ value, label: optionLabel, description }) => appendDatasetOption(value, optionLabel, group, description));
	els.datasetSelect.appendChild(group);
}

/** Bind event listeners to all controls, updating the state and re-rendering the table whenever a control value changes. */
function bindControls() {
	if (els.benchmarkModeSelect) {
		els.benchmarkModeSelect.addEventListener("change", async (event) => {
			await switchBenchmarkMode(event.target.value);
		});
	}

	if (els.sourceSelect) {
		els.sourceSelect.addEventListener('change', async (event) => {
			await switchSource(event.target.value);
		});
	}

	els.datasetSelect.addEventListener('change', (event) => {
		state.currentDataset = event.target.value;
		state.page = 1;
		render();
	});

	els.itersSelect.addEventListener('change', (event) => {
		state.currentIters = event.target.value;
		state.page = 1;
		render();
	});

	els.evaluationCaseSelect.addEventListener('change', (event) => {
		state.mode = event.target.value;
		state.page = 1;
		render();
	});

	els.solverVariantSelect.addEventListener('change', (event) => {
		state.roVariant = event.target.value;
		state.page = 1;
		render();
	});

	els.depthTypeSelect.addEventListener('change', (event) => {
		state.depthType = event.target.value;
		state.page = 1;
		render();
	});

	els.searchInput.addEventListener('input', (event) => {
		state.search = event.target.value;
		state.page = 1;
		render();
	});

	els.pageSizeSelect.addEventListener('change', (event) => {
		state.pageSize = Number(event.target.value);
		state.page = 1;
		render();
	});

	els.hideGtOnly.addEventListener('change', (event) => {
		state.hideGtOnly = event.target.checked;
		state.page = 1;
		render();
	});

	els.bestMdeOnly.addEventListener('change', (event) => {
		state.bestMdeOnly = event.target.checked;
		state.page = 1;
		render();
	});

	els.prevPageBtn.addEventListener('click', () => {
		if (state.page > 1) {
			state.page -= 1;
			render();
		}
	});

	els.nextPageBtn.addEventListener('click', () => {
		state.page += 1;
		render();
	});
}


/** Switch between standard and group-based CSV sources. */
async function switchBenchmarkMode(mode) {
	if (!BENCHMARK_CONFIG[mode] || mode === state.benchmarkMode) return;

	state.benchmarkMode = mode;
	state.currentSource = Object.keys(BENCHMARK_CONFIG[mode].sources)[0];
	state.currentDataset = 'mean';
	state.currentIters = '1000';
	state.page = 1;

	const bmLabel = tLabel(`controls.select.benchmark.${mode}`);
	setTableLoadingState({ key: 'table.loading', params: { option: bmLabel } });

	try {
		await loadData();
		populateControls();
		render();
	}
	catch (error) {
		console.error(error);
		setTableErrorState({ key: 'table.loadFail' });
	}
}

/** Switch the CSV source within the current benchmark mode. */
async function switchSource(sourceKey) {
	const sources = BENCHMARK_CONFIG[state.benchmarkMode].sources;
	if (!sources[sourceKey] || sourceKey === state.currentSource) return;

	state.currentSource = sourceKey;
	state.currentDataset = 'mean';
	state.currentIters = '1000';
	state.page = 1;

	setTableLoadingState({ key: 'table.loading', params: { option: tLabel(sources[sourceKey].key) } });

	try {
		await loadData();
		populateControls();
		render();
	}
	catch (error) {
		console.error(error);
		setTableErrorState({ key: 'table.loadFail' });
	}
}


/** Main render function that processes the rows based on current state, renders the table head, body, and summary. */
function render() {
	const tableRows = () => {
		let rows = [...state.rawRows];
		rows = applyMode(rows);
		rows = applyRoVariant(rows);
		rows = applyDepthType(rows);
		rows = applyIters(rows);
		rows = applyDatasetOrMean(rows);

		if (state.hideGtOnly) rows = rows.filter((row) => !isGtMde(row.mde));

		rows = applySearch(rows);
		rows = applyBestMdeOnly(rows);
		rows = sortRows(rows);
		return rows;
	};

	renderTableHead();
	renderTable(tableRows());
	renderSummary(tableRows());
}


/** Filter rows based on the selected mode, which determines which solvers to include. */
function applyMode(rows) {
	const cfg = MODE_CONFIG[state.mode];

	if (state.mode === 'all') return rows;
	return rows.filter((row) => {
		const solverOk = cfg.solvers.has(baseSolverName(row.solver));
		if (!solverOk) return false;
		if (state.mode === 'uncalibrated' && isCalibMde(row.mde)) return false;

		return true;
	});
}

/** Filter rows based on the selected 'ro' variant, either including only 'ro' solvers or excluding them. */
function applyRoVariant(rows) {
	if (state.roVariant === 'all') return rows;
	return rows.filter((row) => {
		if (state.roVariant === 'ro') return isRoSolver(row.solver);
		return !isRoSolver(row.solver);
	});
}

/** Filter rows based on the selected depth type. */
function applyDepthType(rows) {
	if (state.depthType === 'all') return rows;
	return rows.filter((row) => {
		if (state.depthType === 'affine') return isAISolver(row.solver);
		else return !isAISolver(row.solver);
	});
}

/** Filter rows based on the selected number of iterations, or return all rows if 'all' is selected. */
function applyIters(rows) {
	if (state.currentIters === 'all') return rows;
	return rows.filter((row) => String(row.iters) === String(state.currentIters));
}

/** Apply the selected dataset/group/scene scope, aggregating when a mean option is selected. */
function applyDatasetOrMean(rows) {
	const prefix = tLabel('controls.select.dataset.meanOverGroupPrefix');

	if (state.benchmarkMode !== 'group') {
		if (state.currentDataset !== 'mean') {
			return rows.filter((row) => String(row.dataset) === state.currentDataset);
		}

		return aggregateMeanRows(rows, ['dataset'], { dataset: 'Mean' });
	}

	if (state.currentDataset === 'mean') {
		return aggregateMeanRows(rows, ['dataset', 'group'], { dataset: 'Mean', group: 'Mean' });
	}

	if (state.currentDataset.startsWith('group:')) {
		const groupName = state.currentDataset.slice('group:'.length);
		const groupRows = rows.filter((row) => String(row.group) === groupName);

		return aggregateMeanRows(groupRows, ['dataset', 'group'], {
			dataset: `${prefix} ${csvValueLabel('group', groupName)}`, group: groupName
		});
	}

	if (state.currentDataset.startsWith('scene:')) {
		const sceneName = state.currentDataset.slice('scene:'.length);
		return rows.filter((row) => String(row.dataset) === sceneName);
	}

	return rows.filter((row) => String(row.dataset) === state.currentDataset);
}

/** Aggregate numeric metrics over a selected scope while preserving experiment identity columns. */
function aggregateMeanRows(rows, excludedGroupColumns, baseOverrides = {}) {
	const numericColumns = state.columns.filter((column) => state.numericColumns.has(column) && column !== 'iters');
	const groupColumns = state.columns.filter((column) => (!numericColumns.includes(column) && !excludedGroupColumns.includes(column)));
	const groups = new Map();

	for (const row of rows) {
		const key = groupColumns.map((column) => String(row[column] ?? '')).join('||');

		if (!groups.has(key)) {
			groups.set(key, {
				count: 0,
				sums: Object.fromEntries(numericColumns.map((column) => [column, 0])),
				base: Object.fromEntries(groupColumns.map((column) => [column, row[column]]))
			});
		}

		const group = groups.get(key);
		group.count += 1;

		numericColumns.forEach((column) => {
			const value = Number(row[column]);
			if (!Number.isNaN(value)) group.sums[column] += value;
		});
	}

	const aggregated = [];

	for (const group of groups.values()) {
		const row = { ...baseOverrides, ...group.base };

		numericColumns.forEach((column) => {
			row[column] = group.count ? group.sums[column] / group.count : null;
		});

		aggregated.push(row);
	}

	return aggregated;
}

/** Filter rows based on the search query, checking if any of the specified columns contain the query string. */
function applySearch(rows) {
	const query = state.search.trim().toLowerCase();
	if (!query) return rows;
	return rows.filter((row) => state.columns.some((column) => normalizeForSearch(row[column]).includes(query)));
}

/** Filter rows to include only the best MDE for each MDE family, based on pose_mAA_10, mean_inliers, and mean_mde_runtime metrics. */
function applyBestMdeOnly(rows) {
	if (!state.bestMdeOnly) return rows;

	const bestRows = new Map();

	for (const row of rows) {
		const key = getMdeFamilyKey(row.mde);
		const previous = bestRows.get(key);

		if (!previous || isBetterBestMdeCandidate(row, previous)) {
			bestRows.set(key, row);
		}
	}

	return [...bestRows.values()];
}

/** Sort rows based on the current sort key and direction, handling both numeric and string values appropriately. */
function sortRows(rows) {
	const direction = state.sortDir === 'asc' ? 1 : (state.sortDir === 'desc' ? -1 : 0);

	if (!state.sortKey || direction === 0) return rows;
	else return [...rows].sort((a, b) => {
		const aValue = a[state.sortKey];
		const bValue = b[state.sortKey];
		const aIsNumber = typeof aValue === 'number';
		const bIsNumber = typeof bValue === 'number';

		if (aIsNumber && bIsNumber) {
			if (aValue === bValue) return 0;
			return (aValue - bValue) * direction;
		}

		return String(aValue ?? '').localeCompare(String(bValue ?? ''), undefined, { numeric: true, sensitivity: 'base' }) * direction;
	});
}


/** Determine if a candidate row is a better choice for the best MDE than the current row, based on pose_mAA_10, mean_inliers, and mean_mde_runtime. */
function isBetterBestMdeCandidate(candidate, current) {
	const candidateScore = Number(candidate.pose_mAA_10 ?? Number.NEGATIVE_INFINITY);
	const currentScore = Number(current.pose_mAA_10 ?? Number.NEGATIVE_INFINITY);

	if (candidateScore !== currentScore) {
		return candidateScore > currentScore;
	}

	const candidateInliers = Number(candidate.mean_inliers ?? Number.NEGATIVE_INFINITY);
	const currentInliers = Number(current.mean_inliers ?? Number.NEGATIVE_INFINITY);

	if (candidateInliers !== currentInliers) {
		return candidateInliers > currentInliers;
	}

	const candidateRuntime = Number(candidate.mean_mde_runtime ?? Number.POSITIVE_INFINITY);
	const currentRuntime = Number(current.mean_mde_runtime ?? Number.POSITIVE_INFINITY);

	return candidateRuntime < currentRuntime;
}


/** Render the table head with sortable column headers, indicating the current sort column and direction. */
function renderTableHead() {
	const visibleColumns = state.columns.filter((column) => isColumnVisible(column));

	els.resultsHead.innerHTML = `
		<tr>
			${visibleColumns.map((column) => {
				const isActive = state.sortKey === column;
				const icon = isActive ? (state.sortDir === 'asc' ? 'pi-sort-up' : 'pi-sort-down') : 'pi-sort-alt';
				const dict = t(`table.column.${column}`);
				const titleAttr = getTitleAttr(`table.column.${column}`);
				
				return `
					<th>
						<button class="sort-button ${isActive ? 'is-active' : ''}" type="button" data-column="${column}" ${titleAttr}>
							<span>${dict?.label || column}</span>
							<i class="pi ${icon}"></i>
						</button>
					</th>
				`;
			}).join('')}
		</tr>
	`;

	els.resultsHead.querySelectorAll('[data-column]').forEach((button) => {
		button.addEventListener('click', () => {
			const column = button.dataset.column;

			if (state.sortKey === column) {
				if (state.sortDir === 'desc') state.sortDir = 'asc';
				else if (state.sortDir === 'asc') {
					state.sortDir = null;
					state.sortKey = null;
				}
			}
			else {
				state.sortKey = column;
				state.sortDir = 'desc';
			}

			state.page = 1;
			render();
		});
	});
}

/** Render the table body with the current page of rows, applying special styling for top-ranked pose_mAA_10 values. */
function renderTable(rows) {
	const totalRows = rows.length;
	const totalPages = Math.max(1, Math.ceil(totalRows / state.pageSize));
	state.page = Math.min(state.page, totalPages);

	if (!totalRows) {
		state.visibleRows = [];
		els.resultsBody.innerHTML = '';
		els.emptyState.hidden = false;
		els.emptyState.textContent = tLabel('table.empty');
		setOptionTitle(els.emptyState, tDesc('table.empty'));
		els.paginationBar.hidden = true;
		return;
	}

	const topRanks = getTopPoseRanks(rows);
	const start = (state.page - 1) * state.pageSize;
	const pageRows = rows.slice(start, start + state.pageSize);
	state.visibleRows = pageRows;

	els.emptyState.hidden = true;
	els.paginationBar.hidden = false;
	els.paginationInfo.textContent = tLabel('pagination.info', {
		from: start + 1, to: Math.min(start + state.pageSize, totalRows), total: totalRows, page: state.page, totalPages: totalPages
	});
	setOptionTitle(els.paginationInfo, tDesc('pagination.info'));
	els.prevPageBtn.disabled = state.page <= 1;
	els.nextPageBtn.disabled = state.page >= totalPages;

	els.resultsBody.innerHTML = pageRows.map((row) => {
		const rowId = getRowId(row);
		const topRank = topRanks.get(rowId);
		const visibleColumns = state.columns.filter((column) => isColumnVisible(column));
		
		return `
			<tr class="benchmark-row">
				${visibleColumns.map((column) => {
					const classes = ((column === 'pose_mAA_10' && topRank) ? [`pose-top-${topRank}`] : []);
					const label = csvValueLabel(column, row[column]);
					const desc = csvValueDesc(column, row[column]);
					const titleAttr = desc ? ` title="${desc}"` : '';
					
					return `<td class="${classes.join(' ')}" ${titleAttr}>${label}</td>`;
				}).join('')}
			</tr>
		`;
	}).join('');
}


/** Render the summary section with statistics about the current view, including the best pose_mAA_10 value and counts of datasets, estimators, and solvers. */
function renderSummary(rows) {
	const bm = state.benchmarkMode;
	const datasetsShown = new Set(rows.map((row) => String(row.dataset))).size;
	const estimators = new Set(rows.map((row) => String(row.mde))).size;
	const solvers = new Set(rows.map((row) => String(row.solver))).size;

	const stats = [
		['summary.visibleRows', rows.length],
		...(bm === "standard" ? [['summary.datasetsInView', datasetsShown]] : []),
		['summary.estimators', estimators],
		['summary.solvers', solvers]
	];

	els.summaryGrid.innerHTML = stats.map(([key, value]) => {
		return `
			<div class="summary-box" ${getTitleAttr(key)}>
				<div class="label-title">${tLabel(key)}</div>
				<div class="value">${value}</div>
			</div>
		`;
	}).join('');
}


/** Generate a unique ID for a row based on the values of all columns, used for ranking purposes. */
function getRowId(row) {
	return state.columns.map((column) => `${column}:${String(row[column] ?? '')}`).join('|');
}

/** Determine the top 3 rows based on the 'pose_mAA_10' metric and return a map of row IDs to their rank (1, 2, or 3). */
function getTopPoseRanks(rows) {
	if (!state.columns.includes('pose_mAA_10')) return new Map();
	const ranked = [...rows].sort((a, b) => (b.pose_mAA_10 ?? -Infinity) - (a.pose_mAA_10 ?? -Infinity));
	const rankMap = new Map();
	
	[1, 2, 3].forEach((rank, index) => {
		const row = ranked[index];
		if (row) rankMap.set(getRowId(row), rank);
	});
	return rankMap;
}

/** Check if a column is visible based on the current benchmark mode. */
function isColumnVisible(column) {
	return state.benchmarkMode === 'standard' && column === 'group' ? false : true;
}

/** Render a compact loading message while switching table CSV sources. */
function setTableLoadingState(lang) {
	els.resultsHead.innerHTML = '';
	els.resultsBody.innerHTML = '';
	els.emptyState.hidden = false;
	els.paginationBar.hidden = true;
	els.emptyState.textContent = tLabel(lang.key, lang.params);
	setOptionTitle(els.emptyState, tDesc(lang.key, lang.params));
	
}

/** Render a compact error message for CSV load failures. */
function setTableErrorState(lang) {
	els.emptyState.hidden = false;
	els.paginationBar.hidden = true;
	els.emptyState.textContent = tLabel(lang.key, lang.params);
	setOptionTitle(els.emptyState, tDesc(lang.key, lang.params));
}