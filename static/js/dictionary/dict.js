/*
 * Unified text dictionary.
 *
 * Every entry follows the same shape:  { label, description }
 *   - `label`       — the displayed text (may contain inline HTML like <code>)
 *   - `description` — tooltip text/title shown on hover (null = no tooltip)
 */

/** Dictionary of all text entries of type { label: string, description?: string | null } */
const dict = {

	// Hero section

	'hero.title': { label: 'Depth2Pose: A Pose-Based Benchmark for Monocular Depth Estimation without Ground-Truth Depth' },


	// Toggle buttons

	'controls.toggle.show': { label: 'Show controls' },
	'controls.toggle.hide': { label: 'Hide controls' },
	'table.toggle.show': { label: 'Show results table' },
	'table.toggle.hide': { label: 'Hide results table' },
	'results.toggle.show': { label: 'Show table summary' },
	'results.toggle.hide': { label: 'Hide table summary' },
	'examples.toggle.show': { label: 'Show image examples' },
	'examples.toggle.hide': { label: 'Hide image examples' },


	// Pagination

	'pagination.prev': { label: 'Previous' },
	'pagination.next': { label: 'Next' },
	'pagination.info': { label: 'Showing {from}\u2013{to} of {total} rows \xB7 page {page} / {totalPages}' },


	// Controls

	'controls.label.benchmark': {
		label: 'Benchmark',
		description: 'Switch between different benchmark evaluation setups.'
	},
	'controls.label.source': {
		label: 'Match type',
		description: 'Select the matching pipeline used to produce correspondences.'
	},
	'controls.label.dataset.standard': {
		label: 'Dataset',
		description: 'Filter results by individual dataset or show the overall mean.'
	},
	'controls.label.dataset.group': {
		label: 'Group / scene',
		description: 'Filter by group mean, individual scene, or the global mean.'
	},
	'controls.label.iters': {
		label: 'Iters',
		description: 'Filter by RANSAC iteration count.'
	},
	'controls.label.evaluationCase': {
		label: 'Evaluation case',
		description: 'Filter by calibration mode.'
	},
	'controls.label.solverVariant': {
		label: 'Solver variant',
		description: 'Choose between standard estimator using hybrid error and reprojection-only estimator.'
	},
	'controls.label.depthType': {
		label: 'Depth type',
		description: 'Choose between scale invariant and affine invariant.'
	},
	'controls.label.search': {
		label: 'Search',
		description: 'Free-text search across all visible columns.'
	},
	'controls.search.placeholder': {
		label: 'DepthAnything, calib_shift, pt, ...',
		description: null
	},
	'controls.label.pageSize': {
		label: 'Page size',
		description: 'Number of rows displayed per page.'
	},
	'controls.label.filters': { label: 'Filters' },

	// Select options: benchmark mode

	'controls.select.benchmark.standard': {
		label: 'Standard benchmark',
		description: 'Per-dataset evaluation using standard matching pipelines.'
	},
	'controls.select.benchmark.group': {
		label: 'D2P Dataset',
		description: 'Evaluation on the dataset proposed in the paper'
	},

	// Select options: match type

	'controls.select.source.loma': {
		label: 'LoMa',
		description: null
	},
	'controls.select.source.splg': {
		label: 'SP+LG',
		description: null
	},

	// Select options: dataset mean labels

	'controls.select.dataset.mean.standard': {
		label: 'Mean over datasets',
		description: 'Average metrics across all datasets.'
	},
	'controls.select.dataset.mean.group': {
		label: 'Mean over all scenes',
		description: 'Average metrics across all scenes and groups.'
	},
	'controls.select.dataset.meanOverGroupPrefix': {
		label: 'Mean over',
		description: null
	},

	// Select options: iters

	'controls.select.iters.all': {
		label: 'All',
		description: 'Show results for all RANSAC iteration counts.'
	},


	// Select options: evaluation case

	'controls.select.evaluationCase.all': {
		label: 'All',
		description: 'Show both calibrated and uncalibrated results.'
	},
	'controls.select.evaluationCase.calibrated': {
		label: 'With calibration',
		description: 'Only estimators that use ground-truth intrinsics.'
	},
	'controls.select.evaluationCase.uncalibrated': {
		label: 'Without calibration',
		description: 'Solvers estimators that assume unknown intrinsics.'
	},

	// Select options: solver variant

	'controls.select.solverVariant.all': {
		label: 'All',
		description: 'Show both hybrid (H) and reprojection-only (R) RANSAC results.'
	},
	'controls.select.solverVariant.non_ro': {
		label: 'Hybrid (H)',
		description: 'Hybrid (H) RANSAC solver variants.'
	},
	'controls.select.solverVariant.ro': {
		label: 'Reprojection-only (R)',
		description: 'RANSAC estimators using only reprojection error for scoring. Uses only those 2D-2D correspondences which are inliers wrt ground truth pose.'
	},

	// Select options: depth type
	'controls.select.depthType.all': {
		label: 'All',
		description: 'Show both scale invariant and affine invariant results.'
	},
	'controls.select.depthType.scale': {
		label: 'Scale-invariant',
		description: 'Show results for the assumption of scale-invariant depth.'
	},
	'controls.select.depthType.affine': {
		label: 'Affine-invariant',
		description: 'Show results for the assumption of affine-invariant depth.'
	},

	// Filter checkboxes

	'controls.filter.hideGtOnly': {
		label: 'Hide <code>gt</code> / oracle rows',
		description: 'Exclude ground-truth / oracle depth entries from the table.'
	},
	'controls.filter.bestMdeOnly': {
		label: 'Show only best row per MDE family',
		description: 'Keep only the row with the highest mAA(10°) within each MDE family.'
	},


	// Table columns

	'table.column.group': {
		label: 'Group',
		description: 'Scene type'
	},
	'table.column.dataset': {
		label: 'Dataset',
		description: 'Name of the evaluation dataset.'
	},
	'table.column.dataset.group': {
		label: 'Scene',
		description: 'Individual scene within a scene type.'
	},
	'table.column.iters': {
		label: 'Iters',
		description: 'Number of RANSAC iterations used during pose estimation.'
	},
	'table.column.mde': {
		label: 'MDE',
		description: 'Monocular depth estimation method.'
	},
	'table.column.solver': {
		label: 'Solver',
		description: 'Relative pose solver used for estimation.'
	},
	'table.column.pose_mAA_10': {
		label: 'mAA(10°)',
		description: 'Mean Average Accuracy at 10° threshold for pose estimation.'
	},
	'table.column.mean_mde_runtime': {
		label: 'MDE Runtime (ms)',
		description: 'Average inference time of the depth estimator in milliseconds.'
	},
	'table.column.mean_inliers': {
		label: 'Inliers (%)',
		description: 'Average percentage of inlier correspondences.'
	},

	// Value aliases (CSV value → display label)
	/*
	 * Use 'table.value.[mde/solver/dataset/scene/group/...].<raw>' to rename CSV values.
	 * Any value without an entry here falls back  to the raw CSV string.
	 *
	 * Example:
	 *   'table.value.mde.mde1': {
	 *       label: 'One Super MDE',
	 *       description: 'A particularly interesting test mde.'
	 *   },
	 */
	
	 'table.value.solver.calib': {
		 label: 'H',
		 description: 'Calibrated solver with the assumption of scale-invariant depth using hybrid error in RANSAC.'
	 },
	 'table.value.solver.calib_ro': {
		 label: 'R',
		 description: 'Calibrated solver with the assumption of scale-invariant depth using only the symmetric reprojection error in RANSAC. Only 2D-2D correspondences which are inliers wrt GT pose are used.'
	 },	
	 'table.value.solver.calib_shift': {
		 label: 'H_s',
		 description: 'Calibrated solver with the assumption of affine-invariant depth using hybrid error in RANSAC.'
	 },
	 'table.value.solver.calib_shift_ro': {
		 label: 'R_s',
		 description: 'Calibrated solver with the assumption of affine-invariant depth using only the symmetric reprojection error in RANSAC. Only 2D-2D correspondences which are inliers wrt GT pose are used.'
	 },
	 'table.value.solver.baseline_calib': {
		 label: 'R',
		 description: 'Standard 5-pt solver for calibrated relative pose using Sampson error.'
	 },
	 
	 
	// Empty / loading states

	'table.loading': { label: 'Loading {option}…' },
	'table.empty': { label: 'No rows match the current filters.' },
	'table.loadFail': { label: 'Failed to load CSV results.' },


	// Result cards

	'results.title': { label: 'Table Summary' },
	'results.note': { label: 'Rows are computed directly in the browser from the selected CSV file.' },

	// Summary cards

	'summary.visibleRows': {
		label: 'Visible rows',
		description: 'Total number of rows matching current filters.'
	},
	'summary.datasetsInView': {
		label: 'Datasets in view',
		description: 'Number of distinct datasets in the current selection.'
	},
	'summary.datasetsInView.group': {
		label: 'Scopes in view',
		description: 'Number of distinct scenes/groups shown.'
	},
	'summary.estimators': {
		label: 'Estimators',
		description: 'Distinct monocular depth estimation methods.'
	},
	'summary.solvers': {
		label: 'Solvers',
		description: 'Distinct relative pose solvers.'
	},


	// Image examples

	'examples.title': { label: 'Image Examples' },
	
	'examples.subtitle.choose-dataset': { label: 'Choose a dataset to browse image pairs.' },
	'examples.subtitle.dataset': { label: 'Dataset: {datasetName}' },
	
	'examples.loading.datasets': { label: 'Loading example datasets…' },
	'examples.loading.pairs': { label: 'Loading image pairs for {datasetName}…' },
	
	'examples.empty.datasets': { label: 'No example datasets are currently available.' },
	'examples.empty.pairs': { label: 'No image pairs are currently available for dataset {datasetName}.' },
	
	'examples.failed.datasets': { label: 'Failed to load example datasets.' },
	'examples.failed.pairs': { label: 'Failed to load image pairs for dataset {datasetName}.' },

	'examples.dataset.detail': { label: 'Select a dataset first, then click any pair to inspect its depth results.' },
	'examples.dataset.detail.count': { label: '{count} pairs in this dataset.' },
	'examples.dataset.detail.open': { label: 'Open pairs list' },
	
	'examples.pairs.detail': { label: 'Pairs are ordered by descending best <code>p_err</code>, from worst to best. Click a pair to open depth comparison.' },
	'examples.pairs.card.detail': { label: 'best p_err {bestPerr} · baseline {baseline}' },
	'examples.pairs.card.inliers': { label: '{inliers} inliers' },
	'examples.pairs.card.outliers': { label: '{outliers} outliers' },
	'examples.pairs.card.unused': { label: '{unused} unused' },
	'examples.pairs.card.showdepth': { label: 'Show depth comparison' },
	'examples.pairs.card.showcorrespondences': { label: 'Show correspondences' },
	'examples.pairs.card.hidecorrespondences': { label: 'Hide correspondences' },

	'examples.depth.not-available': { label: 'Selected image pair is no longer available.' },
	'examples.depth.detail': { label: 'Depth results are ordered by ascending <code>pose error</code>, from best to worst for this pair.</p>' },
	'examples.depth.card.solver': { label: '<strong>Estimator:</strong> {solverName}' },
	'examples.depth.card.perr': { label: '<strong>Pose Error:</strong> {pErr}' },
	'examples.depth.card.baseline': { label: '<strong>baseline:</strong> {baselineErr} {comparison}' },

	'examples.back': { label: 'Back' },
	'examples.back-to-datasets': { label: 'Back to datasets' },
	'examples.back-to-pairs': { label: 'Back to pairs' }

};

/** Get an entry from the dictionary (returns { label, description } or null). */
export function t(key) {
	return dict[key] ?? null;
}

export default dict;