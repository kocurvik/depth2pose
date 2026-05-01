import { escapeHtml } from '../global.js';
import { buildInlierSet, datasetAssetUrl, decoratePairCards, formatMetric, formatSignedMetric,
	getComparableResults, renderPairOverviewCard, shortPairName } from './shared.js';

/* Open the depth comparison workflow for one selected image pair. */
export function openPairDepths(pairKey, state, controller) {
	const example = state.currentExamples.find((item) => item.pairKey === pairKey);

	state.view = 'depths';
	state.currentPairKey = pairKey;

	if (!example) {
		controller.renderPanelShell(
			'Image Examples',
			`Dataset: ${state.currentDatasetName}`,
			'<div class="example-empty-state">Selected image pair is no longer available.</div>',
			{
				showBackButton: true,
				backLabel: 'Back to pairs',
				onBack: () => {
					controller.renderPairListView();
					controller.focusExamplesViewer();
				}
			}
		);
		controller.focusExamplesViewer();
		return;
	}

	renderDepthDetailView(example, state, controller);
}

/* Render the depth detail view from an already resolved image-pair example. */
function renderDepthDetailView(example, state, controller) {
	const preferredResults = getComparableResults(example);
	const overviewResult = example.selectedResult || preferredResults[0]?.result || null;

	controller.renderPanelShell(
		'Image Examples',
		`Dataset: ${state.currentDatasetName}`,
		`
			<div class="pair-detail-header">
				<div>
					<p class="muted-note mb-0">Depth results are ordered by ascending <code>p_err</code>, from best to worst for this pair.</p>
				</div>
			</div>

			<div class="pair-detail-overview">
				${renderPairOverviewCard(example, overviewResult)}
			</div>

			<div class="depth-result-stack">
				${preferredResults.map((item, index) => renderDepthResultCard(item, index, state.currentDatasetName, example)).join('')}
			</div>
		`,
		{
			showBackButton: true,
			backLabel: 'Back to pairs',
			onBack: () => {
				controller.renderPairListView();
				controller.focusExamplesViewer();
			}
		}
	);

	decoratePairCards(controller.getPanelBody());
	controller.focusExamplesViewer();
}

/* Render one depth result card for one MDE on the currently selected pair. */
function renderDepthResultCard(item, index, dataset, example) {
	const { mdeName, result } = item;
	const depth1 = result?.exported_depth_image1_path ? datasetAssetUrl(dataset, result.exported_depth_image1_path) : '';
	const depth2 = result?.exported_depth_image2_path ? datasetAssetUrl(dataset, result.exported_depth_image2_path) : '';

	const totalCount = Array.isArray(example?.details?.kp1) ? example.details.kp1.length : 0;
	const unusedCount = Array.isArray(result?.unused_kps) ? result.unused_kps.length : 0;
	const inlierCount = buildInlierSet(totalCount, result?.inliers).size;
	const outlierCount = Math.max(totalCount - inlierCount - unusedCount, 0);

	return `
		<article class="depth-result-card">
			<div class="depth-result-meta">
				<div class="depth-result-rank">#${index + 1}</div>
				<div class="depth-result-text">
					<h4 class="title is-6 mb-1">${escapeHtml(mdeName)}</h4>
					<p class="depth-result-line mb-0">
						<span><strong>solver:</strong> ${escapeHtml(String(result?.solver ?? '—'))}</span>
						<span><strong>p_err:</strong> ${formatMetric(result?.p_err)}</span>
						${renderBaselineComparison(example, result)}
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

/* Render per-depth comparison against the pair baseline p_err. */
function renderBaselineComparison(example, result) {
	const baselineErr = Number(example?.details?.baseline_p_err);
	const pErr = Number(result?.p_err);

	if (!Number.isFinite(baselineErr)) return '';

	const comparison = Number.isFinite(pErr)
		? ` · Δ ${formatSignedMetric(pErr - baselineErr)}`
		: '';

	return `<span><strong>baseline:</strong> ${formatMetric(baselineErr)}${comparison}</span>`;
}