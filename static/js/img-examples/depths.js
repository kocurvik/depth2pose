import { buildInlierSet, datasetAssetUrl, decoratePairCards, formatMetric, formatSignedMetric,
	getComparableResults, renderPairOverviewCard, shortPairName } from './shared.js';
import { tLabel, csvValueLabel, getTitleAttr } from '../dictionary/index.js';
import { t } from '../dictionary/dict.js';


/* Open the depth comparison workflow for one selected image pair. */
export function openPairDepths(pairKey, state, controller) {
	const example = state.currentExamples.find((item) => item.pairKey === pairKey);

	state.view = 'depths';
	state.currentPairKey = pairKey;
	const datasetLang = csvValueLabel('dataset', state.currentDatasetName);

	if (!example) {
		controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
			`<div class="example-empty-state" ${getTitleAttr('examples.depth.not-available')}>
				${tLabel('examples.depth.not-available')}
			</div>`,
			{
				showBackButton: true,
				backLabel: { key: 'examples.back-to-pairs' },
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
	const datasetLang = csvValueLabel('dataset', state.currentDatasetName);

	controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
		`
			<div class="pair-detail-header">
				<div>
					<p class="muted-note mb-0" ${getTitleAttr('examples.depth.detail')}>
						${tLabel('examples.depth.detail')}
					</p>
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
			backLabel: { key: 'examples.back-to-pairs' },
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

	const solverLabel = t('table.value.solver.' + String(result?.solver ?? '—'))?.label || String(result?.solver ?? '—');
	const mdeLabel = t('table.value.mde.' + String(mdeName ?? '—'))?.label || String(mdeName ?? '—');

	return `
		<article class="depth-result-card">
			<div class="depth-result-meta">
				<div class="depth-result-rank">#${index + 1}</div>
				<div class="depth-result-text">
					<h4 class="title is-6 mb-1">${mdeLabel}</h4>
					<p class="depth-result-line mb-0">
						<span ${getTitleAttr('examples.depth.card.solver', { solverName: solverLabel })}>
							${tLabel('examples.depth.card.solver', { solverName: solverLabel })}
						</span>
						<span ${getTitleAttr('examples.depth.card.perr', { pErr: formatMetric(result?.p_err) })}>
							${tLabel('examples.depth.card.perr', { pErr: formatMetric(result?.p_err) })}
						</span>
						${renderBaselineComparison(example, result)}
					</p>
				</div>
			</div>

			<div class="example-stat-pills example-stat-pills-overview">
				<span class="example-stat-pill is-inlier" ${getTitleAttr('examples.pairs.card.inliers', { inliers: inlierCount })}>
					${tLabel('examples.pairs.card.inliers', { inliers: inlierCount })}
				</span>
				<span class="example-stat-pill is-outlier" ${getTitleAttr('examples.pairs.card.outliers', { outliers: outlierCount })}>
					${tLabel('examples.pairs.card.outliers', { outliers: outlierCount })}
				</span>
				<span class="example-stat-pill is-unused" ${getTitleAttr('examples.pairs.card.unused', { unused: unusedCount })}>
					${tLabel('examples.pairs.card.unused', { unused: unusedCount })}
				</span>
			</div>

			<div class="example-depth-grid">
				${depth1 ? `<figure class="example-depth-panel"><img src="${depth1}" alt="Depth map 1 for ${mdeLabel}"><figcaption>Depth 1 · ${mdeLabel}</figcaption></figure>` : '<div class="example-depth-panel example-depth-empty">Depth 1 not available</div>'}
				${depth2 ? `<figure class="example-depth-panel"><img src="${depth2}" alt="Depth map 2 for ${mdeLabel}"><figcaption>Depth 2 · ${mdeLabel}</figcaption></figure>` : '<div class="example-depth-panel example-depth-empty">Depth 2 not available</div>'}
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
		? `· Δ ${formatSignedMetric(pErr - baselineErr)}`
		: '';

	return `
		<span ${getTitleAttr('examples.depth.card.baseline', { baselineErr: formatMetric(baselineErr), comparison })}>
			${tLabel('examples.depth.card.baseline', { baselineErr: formatMetric(baselineErr), comparison })}
		</span>
	`;
}