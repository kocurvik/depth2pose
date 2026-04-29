const CSV_URL = 'https://raw.githubusercontent.com/lbujnak/depth2pose_webdata/main/benchmark/new/slim_pose_results.csv';

const MODE_CONFIG = {
	calibrated: {
		label: 'With calibration',
		solvers: new Set(['calib', 'calib_shift', 'baseline_calib'])
	},
	uncalibrated: {
		label: 'Without calibration',
		solvers: new Set([
			'sf', 'sf_shift',
			'vf', 'vf_shift',
			'mdecalib', 'mdecalib_shift',
			'baseline_sf', 'baseline_vf'
		])
	}
};

const state = {
	rawRows: [],
	columns: [],
	numericColumns: new Set(),
	datasets: [],
	itersOptions: [],
	currentDataset: 'mean',
	currentIters: 'all',
	search: '',
	mode: 'calibrated',
	roVariant: 'non_ro',
	hideGtOnly: true,
	bestMdeOnly: false,
	sortKey: 'pose_mAA_10',
	sortDir: 'desc',
	page: 1,
	pageSize: 25,
	visibleRows: []
};

const els = {
	datasetSelect: document.getElementById('datasetSelect'),
	itersSelect: document.getElementById('itersSelect'),
	evaluationCaseSelect: document.getElementById('evaluationCaseSelect'),
	solverVariantSelect: document.getElementById('solverVariantSelect'),
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


/* Attach toggle functionality to a card, allowing it to collapse/expand when the button is clicked. */
function attachCardToggle(cardId, buttonId, contentId, showText, hideText) {
	const card = document.getElementById(cardId);
	const button = document.getElementById(buttonId);
	const content = document.getElementById(contentId);
	if (!card || !button || !content) return;

	button.addEventListener('click', () => {
		const collapsed = card.classList.toggle('is-collapsed');
		button.textContent = collapsed ? showText : hideText;
		button.setAttribute('aria-expanded', String(!collapsed));
	});
}

/* Load CSV data from the specified URL, parse it, and populate the state with columns, rows, and options for controls. */
async function loadData() {
	let csvText;

	try {
		const response = await fetch(CSV_URL);
		if (!response.ok) throw new Error(`Failed to load CSV from ${CSV_URL}: ${response.status}`);
		
		csvText = await response.text();
	}
	catch (error) {
		throw new Error('Failed to load CSV.');
	}

	const { columns, rows } = parseCsv(csvText);

	state.columns = columns;
	state.rawRows = rows;
	state.numericColumns = new Set(columns.filter((column) => rows.every((row) => typeof row[column] === 'number' || row[column] === '')));
	state.datasets = [...new Set(rows.map((row) => String(row.dataset)))].sort((a, b) => a.localeCompare(b));
	state.itersOptions = [...new Set(rows.map((row) => String(row.iters)))].sort((a, b) => Number(a) - Number(b));
}

/* Populate the dataset and iters dropdowns based on the loaded data, and set initial values for all controls. */
function populateControls() {
	els.datasetSelect.innerHTML = '';
	[
		{ value: 'mean', label: 'Mean over datasets' }, ...state.datasets.map((dataset) => ({ value: dataset, label: dataset }))
	].forEach(({ value, label }) => {
		const option = document.createElement('option');

		option.value = value;
		option.textContent = label;
		els.datasetSelect.appendChild(option);
	});

	els.itersSelect.innerHTML = '';
	[
		{ value: 'all', label: 'All' }, ...state.itersOptions.map((iters) => ({ value: iters, label: iters }))
	].forEach(({ value, label }) => {
		const option = document.createElement('option');

		option.value = value;
		option.textContent = label;
		els.itersSelect.appendChild(option);
	});

	els.evaluationCaseSelect.innerHTML = '';
	[
		{ value: 'all', label: 'All' },
		{ value: 'calibrated', label: 'With calibration' },
		{ value: 'uncalibrated', label: 'Without calibration' }
	].forEach(({ value, label }) => {
		const option = document.createElement('option');

		option.value = value;
		option.textContent = label;
		els.evaluationCaseSelect.appendChild(option);
	});

	els.solverVariantSelect.innerHTML = '';
	[
		{ value: 'all', label: 'All' },
		{ value: 'non_ro', label: 'Standard RANSAC' },
		{ value: 'ro', label: 'Reprojection-only RANSAC (_ro)' }
	].forEach(({ value, label }) => {
		const option = document.createElement('option');

		option.value = value;
		option.textContent = label;
		els.solverVariantSelect.appendChild(option);
	});

	els.datasetSelect.value = state.currentDataset;
	els.itersSelect.value = state.currentIters;
	els.evaluationCaseSelect.value = state.mode;
	els.solverVariantSelect.value = state.roVariant;
	els.pageSizeSelect.value = String(state.pageSize);
	els.hideGtOnly.checked = state.hideGtOnly;
	els.bestMdeOnly.checked = state.bestMdeOnly;
}

/* Bind event listeners to all controls, updating the state and re-rendering the table whenever a control value changes. */
function bindControls() {
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

	els.resultsBody.addEventListener('click', (event) => {
		const rowEl = event.target.closest('tr[data-row-index]');
		if (!rowEl) return;

		const rowIndex = Number(rowEl.dataset.rowIndex);
		const row = state.visibleRows[rowIndex];
		if (!row) return;

		document.dispatchEvent(new CustomEvent('benchmark:row-click', {
			detail: {
				row: { ...row },
				currentDataset: state.currentDataset,
				currentIters: state.currentIters,
				mode: state.mode,
				roVariant: state.roVariant,
				availableDatasets: [...state.datasets]
			}
		}));
	});
}


/* Parses a CSV string into columns and rows. */
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

/* Parses a single line of CSV, handling quoted values and commas. */
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



/* Main render function that processes the rows based on current state, renders the table head, body, and summary. */
function render() {
	const tableRows = () => {
		let rows = [...state.rawRows];
		rows = applyMode(rows);
		rows = applyRoVariant(rows);
		rows = applyIters(rows);
		rows = applyDatasetOrMean(rows);

		if (state.hideGtOnly) rows = rows.filter((row) => !isGtRow(row));

		rows = applySearch(rows);
		rows = applyBestMdeOnly(rows);
		rows = sortRows(rows);
		return rows;
	};

	renderTableHead();
	renderTable(tableRows());
	renderSummary(tableRows());
}



/* Filter rows based on the selected mode, which determines which solvers to include. */
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

/* Filter rows based on the selected 'ro' variant, either including only 'ro' solvers or excluding them. */
function applyRoVariant(rows) {
	if (state.roVariant === 'all') return rows;
	return rows.filter((row) => {
		if (state.roVariant === 'ro') return isRoSolver(row.solver);
		return !isRoSolver(row.solver);
	});
}

/* Filter rows based on the selected number of iterations, or return all rows if 'all' is selected. */
function applyIters(rows) {
	if (state.currentIters === 'all') return rows;
	return rows.filter((row) => String(row.iters) === String(state.currentIters));
}

/* Either filter rows by the selected dataset or compute the mean across datasets for each unique combination of other parameters. */
function applyDatasetOrMean(rows) {
	if (state.currentDataset !== 'mean') {
		return rows.filter((row) => String(row.dataset) === state.currentDataset);
	}

	const numericColumns = state.columns.filter((column) => state.numericColumns.has(column) && column !== 'iters');
	const groupColumns = state.columns.filter((column) => !numericColumns.includes(column) && column !== 'dataset');
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
		const row = { dataset: 'Mean', ...group.base };

		numericColumns.forEach((column) => {
			row[column] = group.count ? group.sums[column] / group.count : null;
		});

		aggregated.push(row);
	}
	return aggregated;
}

/* Filter rows based on the search query, checking if any of the specified columns contain the query string. */
function applySearch(rows) {
	const query = state.search.trim().toLowerCase();
	if (!query) return rows;
	return rows.filter((row) => state.columns.some((column) => normalizeForSearch(row[column]).includes(query)));
}

/* Filter rows to include only the best MDE for each MDE family, based on pose_mAA_10, mean_inliers, and mean_mde_runtime metrics. */
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

/* Sort rows based on the current sort key and direction, handling both numeric and string values appropriately. */
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


/* Extract the base solver name by removing any '_ro' suffix, used for grouping related solvers together. */
function baseSolverName(solver) {
	return String(solver ?? '').replace(/_ro$/, '');
}

/* Extract the MDE family key by taking the prefix before any '-' character and removing 'Calib', used for grouping related MDEs together. */
function getMdeFamilyKey(mde) {
	const prefix = String(mde ?? '').split('-')[0];
	return prefix.replaceAll('Calib', '');
}

/* Check if a solver name indicates that it is a 'ro' variant, based on the presence of the '_ro' suffix. */
function isRoSolver(solver) {
	return String(solver ?? '').endsWith('_ro');
}

/* Check if an MDE name indicates that it is a calibrated method, based on the presence of 'Calib' in the name. */
function isCalibMde(mde) {
	return String(mde ?? '').includes('Calib');
}

/* Determine if a candidate row is a better choice for the best MDE than the current row, based on pose_mAA_10, mean_inliers, and mean_mde_runtime. */
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



/* Render the table head with sortable column headers, indicating the current sort column and direction. */
function renderTableHead() {
	els.resultsHead.innerHTML = `
		<tr>
			${state.columns.map((column) => {
				const isActive = state.sortKey === column;
				const icon = isActive ? (state.sortDir === 'asc' ? 'pi-sort-up' : 'pi-sort-down') : 'pi-sort-alt';

				return `
					<th>
						<button class="sort-button ${isActive ? 'is-active' : ''}" type="button" data-column="${column}">
							<span>${column}</span>
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

/* Render the table body with the current page of rows, applying special styling for top-ranked pose_mAA_10 values. */
function renderTable(rows) {
	const totalRows = rows.length;
	const totalPages = Math.max(1, Math.ceil(totalRows / state.pageSize));
	state.page = Math.min(state.page, totalPages);

	if (!totalRows) {
		state.visibleRows = [];
		els.resultsBody.innerHTML = '';
		els.emptyState.hidden = false;
		els.emptyState.textContent = 'No rows match the current filters.';
		els.paginationBar.hidden = true;
		return;
	}

	const topRanks = getTopPoseRanks(rows);
	const start = (state.page - 1) * state.pageSize;
	const pageRows = rows.slice(start, start + state.pageSize);
	state.visibleRows = pageRows;

	els.emptyState.hidden = true;
	els.paginationBar.hidden = false;
	els.paginationInfo.textContent = `Showing ${start + 1}–${Math.min(start + state.pageSize, totalRows)} of ${totalRows} rows · page ${state.page} / ${totalPages}`;
	els.prevPageBtn.disabled = state.page <= 1;
	els.nextPageBtn.disabled = state.page >= totalPages;

	els.resultsBody.innerHTML = pageRows.map((row, rowIndex) => {
		const rowId = getRowId(row);
		const topRank = topRanks.get(rowId);
		
		return `
			<tr class="benchmark-row is-clickable" data-row-index="${rowIndex}">
				${state.columns.map((column) => {
					const classes = [];
					if (column === 'pose_mAA_10' && topRank) classes.push(`pose-top-${topRank}`);
					const value = formatValue(row[column]);
					return `<td class="${classes.join(' ')}" title="${String(value).replace(/"/g, '&quot;')}">${value}</td>`;
				}).join('')}
			</tr>
		`;
	}).join('');
}

/* Render the summary section with statistics about the current view, including the best pose_mAA_10 value and counts of datasets, estimators, and solvers. */
function renderSummary(rows) {
	const datasetsShown = new Set(rows.map((row) => String(row.dataset))).size;
	const estimators = new Set(rows.map((row) => String(row.mde))).size;
	const solvers = new Set(rows.map((row) => String(row.solver))).size;
	const stats = [
		['Visible rows', rows.length],
		['Datasets in view', datasetsShown],
		['Estimators', estimators],
		['Solvers', solvers]
	];

	els.summaryGrid.innerHTML = stats.map(([label, value]) => `
		<div class="summary-box">
			<div class="label-title">${label}</div>
			<div class="value">${value}</div>
		</div>
	`).join('');
}



/* Generate a unique ID for a row based on the values of all columns, used for ranking purposes. */
function getRowId(row) {
	return state.columns.map((column) => `${column}:${String(row[column] ?? '')}`).join('|');
}

/* Determine the top 3 rows based on the 'pose_mAA_10' metric and return a map of row IDs to their rank (1, 2, or 3). */
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



/* Utility function to check if a value is numeric. */
function isNumericValue(value) {
	if (value === '' || value === null || value === undefined) return false;
	return !Number.isNaN(Number(value));
}

/* Formats a value for display, handling null, undefined, and numeric values. */
function formatValue(value) {
	if (value === null || value === undefined || value === '') return '—';
	if (typeof value === 'number') {
		if (Number.isInteger(value)) return String(value);
		return value.toFixed(6).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
	}
	return String(value);
}

/* Normalize a value for search by converting it to a lowercase string, treating null and undefined as empty strings. */
function normalizeForSearch(value) {
	return String(value ?? '').toLowerCase();
}

/* Check if a row corresponds to a ground truth entry based on the 'mde' column value. */
function isGtRow(row) {
	const mdeValue = normalizeForSearch(row.mde);
	return mdeValue === 'gt' || mdeValue === 'ground-truth' || mdeValue === 'ground_truth';
}



/* Initialize the benchmark viewer */
export async function init() {
	attachCardToggle('controlsCard', 'controlsToggle', 'controlsContent', 'Show controls', 'Hide controls');
	attachCardToggle('tableCard', 'tableToggle', 'tableContent', 'Show table', 'Hide table');
	attachCardToggle('resultsCard', 'resultsToggle', 'resultsContent', 'Show results', 'Hide results');

	try {
		await loadData();
		populateControls();
		bindControls();
		render();
	}
	catch (error) {
		console.error(error);
		els.emptyState.hidden = false;
		els.emptyState.textContent = 'Failed to load CSV results.';
		els.paginationBar.hidden = true;
	}
}