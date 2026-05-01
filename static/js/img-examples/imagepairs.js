import { decoratePairCards, getPoseError, renderPairCard } from './shared.js';

/* Render the list of RGB pairs for the currently selected dataset. */
export function renderPairListView(state, controller) {
	const examples = [...state.currentExamples].sort((a, b) => getPoseError(b.selectedResult) - getPoseError(a.selectedResult));

	controller.renderPanelShell(
		'Image Examples',
		`Dataset: ${state.currentDatasetName}`,
		`
			<p class="muted-note mb-4">Pairs are ordered by descending best <code>p_err</code>, from worst to best. Click a pair to open depth comparison.</p>
			<div class="example-card-grid">
				${examples.map((example) => renderPairCard(example)).join('')}
			</div>
		`,
		{
			showBackButton: true,
			backLabel: 'Back to datasets',
			onBack: () => {
				controller.renderDatasetChooserView();
				controller.focusExamplesViewer();
			}
		}
	);

	decoratePairCards(controller.getPanelBody());
	controller.focusExamplesViewer();
}