import { decoratePairCards, getPoseError, renderPairCard } from './shared.js';
import { tLabel, csvValueLabel, getTitleAttr } from '../dictionary/index.js';

/** Render the list of RGB pairs for the currently selected dataset. */
export function renderPairListView(state, controller) {
	const examples = [...state.currentExamples].sort((a, b) => getPoseError(b.selectedResult) - getPoseError(a.selectedResult));
	const datasetLang = csvValueLabel('dataset', state.currentDatasetName);

	controller.renderPanelShell({ key: 'examples.title' }, { key: 'examples.subtitle.dataset', params: { datasetName: datasetLang } },
		`
			<p class="muted-note mb-4" ${getTitleAttr('examples.pairs.detail')}>
				${tLabel('examples.pairs.detail')}
			</p>
			<div class="example-card-grid">
				${examples.map((example) => renderPairCard(example)).join('')}
			</div>
		`,
		{
			showBackButton: true,
			backLabel: { key: 'examples.back-to-datasets' },
			onBack: () => {
				controller.renderDatasetChooserView();
				controller.focusExamplesViewer();
			}
		}
	);

	decoratePairCards(controller.getPanelBody());
	controller.focusExamplesViewer();
}