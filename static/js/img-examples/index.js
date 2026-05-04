import { attachCardToggle } from '../global.js';
import { tLabel, tDesc, setOptionTitle } from '../dictionary/index.js';

import { openPairDepths } from './depths.js';
import { renderPairListView } from './imagepairs.js';
import { loadAvailableDatasets, openDatasetPairList, renderDatasetChooserView } from './datasets.js';

let panelEls = null;

const examplesState = {
	availableDatasets: [],
	datasetSummaries: [],
	currentDatasetName: null,
	currentExamples: [],
	view: 'datasets',
	backAction: null
};

const controller = {
	renderPanelShell, focusExamplesViewer,
	getPanelBody: () => { return ensureExamplesPanel().body || document; },

	openDatasetPairList: (datasetName) => openDatasetPairList(datasetName, examplesState, controller),
	openPairDepths: (pairKey) => openPairDepths(pairKey, examplesState, controller),

	renderDatasetChooserView: () => renderDatasetChooserView(examplesState, controller),
	renderPairListView: () => renderPairListView(examplesState, controller)
};


/** Initialize the embedded benchmark examples panel below the results table. */
export function initImageExamples() {
	ensureExamplesPanel();
	loadAvailableDatasets(examplesState, controller);
}


/** Create the examples card once and bind it to the static markup from index.html. */
function ensureExamplesPanel() {
	if (panelEls) return panelEls;

	const root = document.getElementById('benchmarkExamplesCard');
	if (!root) throw new Error('Missing #benchmarkExamplesCard element.');

	const toggle = root.querySelector('#examplesToggle');
	const content = root.querySelector('#examplesContent');

	attachCardToggle('benchmarkExamplesCard', 'examplesToggle', 'examplesContent', { key: 'examples.toggle.show' }, { key: 'examples.toggle.hide' });

	if (!root.dataset.examplesClickBound) {
		root.dataset.examplesClickBound = 'true';
		root.addEventListener('click', handleExamplesClick);
	}

	const back = root.querySelector('#benchmarkExamplesBack');
	if (back && !back.dataset.bound) {
		back.dataset.bound = 'true';
		back.addEventListener('click', () => {
			if (typeof examplesState.backAction === 'function') examplesState.backAction();
		});
	}

	panelEls = {
		root, toggle, content, back,
		title: root.querySelector('#benchmarkExamplesTitle'),
		subtitle: root.querySelector('#benchmarkExamplesSubtitle'),
		body: root.querySelector('#benchmarkExamplesContent')
	};

	return panelEls;
}

/** Handle embedded panel click actions. */
function handleExamplesClick(event) {
	const datasetButton = event.target.closest('[data-open-dataset]');
	if (datasetButton) {
		const datasetName = datasetButton.getAttribute('data-open-dataset');
		if (datasetName) controller.openDatasetPairList(datasetName);
		return;
	}

	const pairButton = event.target.closest('[data-open-pair]');
	if (pairButton) {
		const pairKey = pairButton.getAttribute('data-open-pair');
		if (pairKey) controller.openPairDepths(pairKey);
	}
}


/** Replace the panel title, subtitle, content, and optional back button state. */
function renderPanelShell(title, subtitle, bodyHtml, options = {}) {
	const els = ensureExamplesPanel();
	const { showBackButton = false, backLabel = { key: 'examples.back' }, onBack = null } = options;

	if (els.title) {
		els.title.textContent = tLabel(title.key, title.params);
		setOptionTitle(els.title, tDesc(title.key, title.params));
	}
	if (els.subtitle) {
		els.subtitle.textContent = tLabel(subtitle.key, subtitle.params);
		setOptionTitle(els.subtitle, tDesc(subtitle.key, subtitle.params));
	}
	if (els.body) els.body.innerHTML = bodyHtml;

	examplesState.backAction = typeof onBack === 'function' ? onBack : null;

	if (els.back) {
		els.back.hidden = !showBackButton;
		els.back.style.display = showBackButton ? 'inline-flex' : 'none';
		
		const label = els.back.querySelector('span:last-child');
		if (label) label.textContent = tLabel(backLabel.key, backLabel.params);
		setOptionTitle(els.back, tDesc(backLabel.key, backLabel.params));
	}
}

/** Smoothly bring the embedded examples viewer into the viewport after view changes. */
function focusExamplesViewer({ resetInnerScroll = true } = {}) {
	const els = ensureExamplesPanel();
	const root = els?.root;
	const body = els?.body;
	if (!root) return;

	requestAnimationFrame(() => {
		if (resetInnerScroll && body) body.scrollTop = 0;
		root.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
	});
}