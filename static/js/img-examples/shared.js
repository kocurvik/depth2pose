import { isGtMde } from '../global.js';
import { IMG_EXAMPLES_SOURCE } from '../api-config.js';
import { tLabel, getTitleAttr, setOptionTitle } from '../dictionary/index.js';

/** Build an absolute URL for one asset inside a dataset example folder. */
export function datasetAssetUrl(dataset, relativePath) {
	return `${IMG_EXAMPLES_SOURCE}${dataset}/${relativePath}`;
}


/** Return all comparable depth results for one pair, ordered best to worst. */
export function getComparableResults(example) {
	return Object.entries(example.details?.results || {})
		.map(([mdeName, result]) => ({ mdeName, result }))
		.filter(({ mdeName, result }) => result && !isGtMde(mdeName))
		.sort((a, b) => {
			const aErr = Number(a.result?.p_err ?? Number.POSITIVE_INFINITY);
			const bErr = Number(b.result?.p_err ?? Number.POSITIVE_INFINITY);
			return aErr - bErr;
		});
}

/** Pick the best comparable depth result for one example details object. */
export function getBestResult(details) {
	return getComparableResults({ details })[0]?.result || null;
}

/** Extract a numeric pose error from a result-like object and return Infinity when missing. */
export function getPoseError(resultLike) {
	return Number(resultLike?.p_err ?? resultLike?.best_mde_p_err ?? Number.POSITIVE_INFINITY);
}


/** Format a numeric metric for compact display in image example cards. */
export function formatMetric(value) {
	const numeric = Number(value);
	if (!Number.isFinite(numeric)) return '—';
	return numeric.toFixed(3).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

/** Format a signed numeric delta for compact comparison against baseline. */
export function formatSignedMetric(value) {
	if (!Number.isFinite(value)) return '—';
	const formatted = formatMetric(Math.abs(value));
	
	if (value === 0) return '0';
	return `${value > 0 ? '+' : '-'}${formatted}`;
}


/** Normalize a pair key so it can be reused as a stable file name fragment. */
export function normalizePairKey(pairKey) {
	return String(pairKey ?? '').replaceAll('\\', '_').replaceAll("'", '_');
}

/** Convert a raw pair key into a shorter human-readable display name. */
export function shortPairName(pairKey) {
	return String(pairKey || '').replaceAll('\\', ' / ').replaceAll("'", ' / ');
}


/** Render the selected pair overview in the same format as pair cards from the list. */
export function renderPairOverviewCard(example, overviewResult) {
	return renderPairCard(example, { selectedResult: overviewResult, includeDepthAction: false, includeCorrespondenceAction: true, extraCardClass: 'pair-detail-example-card', largeVisual: true });
}

/** Render one RGB pair card with summary statistics and correspondence overlay. */
export function renderPairCard(example, options = {}) {
	const { selectedResult = example.selectedResult, includeDepthAction = true, includeCorrespondenceAction = true, extraCardClass = '', largeVisual = false } = options;

	const pErr = selectedResult?.p_err;
	const baselineErr = example.details?.baseline_p_err;
	const unusedCount = Array.isArray(selectedResult?.unused_kps) ? selectedResult.unused_kps.length : 0;
	const totalCount = Array.isArray(example.details?.kp1) ? example.details.kp1.length : 0;
	const inlierCount = buildInlierSet(totalCount, selectedResult?.inliers, selectedResult?.unused_kps).size;
	const outlierCount = Math.max(totalCount - inlierCount - unusedCount, 0);

	const safeExample = encodeURIComponent(JSON.stringify({
		kp1: example.details?.kp1 || [],
		kp2: example.details?.kp2 || [],
		inliers: selectedResult?.inliers || [],
		unusedKps: selectedResult?.unused_kps || []
	}));

	return `
		<article class="example-card ${extraCardClass}">
			<div class="example-card-header">
				<div>
					<h4 class="title is-6 mb-1">${shortPairName(example.pairKey)}</h4>
					<p class="example-meta mb-0" ${getTitleAttr('examples.pairs.card.detail', { bestPerr: formatMetric(pErr), baseline: formatMetric(baselineErr) })}>
						${tLabel('examples.pairs.card.detail', { bestPerr: formatMetric(pErr), baseline: formatMetric(baselineErr) })}
					</p>
				</div>
			</div>

			<div class="example-stat-pills">
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

			<div class="example-visual-pair ${largeVisual ? 'example-visual-pair-large' : ''}" data-example='${safeExample}'>
				<div class="example-image-panel">
					<img class="example-rgb-image" src="${example.rgb1Url}" alt="RGB image 1 for ${example.pairKey}">
					<div class="example-image-label">Image 1</div>
				</div>

				<div class="example-image-panel">
					<img class="example-rgb-image" src="${example.rgb2Url}" alt="RGB image 2 for ${example.pairKey}">
					<div class="example-image-label">Image 2</div>
				</div>

				<svg class="example-correspondence-overlay" aria-hidden="true" hidden></svg>
			</div>

			${(includeDepthAction || includeCorrespondenceAction) ? renderPairCardActions(example, { includeDepthAction, includeCorrespondenceAction }) : ''}
		</article>
	`;
}

/** Render action buttons for one pair card. */
function renderPairCardActions(example, options = {}) {
	const { includeDepthAction = true, includeCorrespondenceAction = true } = options;

	return `
		<div class="example-card-actions">
			${includeCorrespondenceAction ? `
				<button class="button is-light is-small example-correspondence-toggle" type="button" aria-pressed="false" ${getTitleAttr('examples.pairs.card.showcorrespondences')}>
					${tLabel('examples.pairs.card.hidecorrespondences')}
				</button>
			` : ''}

			${includeDepthAction ? `
				<button class="button is-link is-light is-small" type="button" data-open-pair="${example.pairKey}" ${getTitleAttr('examples.pairs.card.showdepth')}>
					${tLabel('examples.pairs.card.showdepth')}
				</button>
			` : ''}
		</div>
	`;
}


/** Decorate every currently visible RGB pair panel with correspondence overlays after images load and after responsive resizes. */
export function decoratePairCards(scope = document) {
	const cards = scope.querySelectorAll('.example-visual-pair[data-example]');

	cards.forEach((card) => {
		if (card.__corrCleanup) card.__corrCleanup();

		const payload = card.getAttribute('data-example');
		if (!payload) return;

		let data;
		try { data = JSON.parse(decodeURIComponent(payload)); }
		catch { return; }

		const imgs = card.querySelectorAll('.example-rgb-image');
		const svg = card.querySelector('.example-correspondence-overlay');
		const parentCard = card.closest('.example-card') || card.parentElement;
		const toggle = parentCard?.querySelector('.example-correspondence-toggle');

		if (imgs.length < 2 || !svg || !toggle) return;

		let isVisible = false;
		let rafId = 0, resizeTimeoutId = 0, lastLayoutKey = '';

		const getLayoutKey = () => {
			const containerRect = card.getBoundingClientRect();
			const rect1 = imgs[0].getBoundingClientRect();
			const rect2 = imgs[1].getBoundingClientRect();

			return [
				Math.round(containerRect.width), Math.round(containerRect.height),
				Math.round(rect1.width), Math.round(rect1.height),
				Math.round(rect2.width), Math.round(rect2.height),
				imgs[0].naturalWidth, imgs[0].naturalHeight,
				imgs[1].naturalWidth, imgs[1].naturalHeight,
			].join(':');
		};

		const canDraw = () => (isVisible && imgs[0].complete && imgs[1].complete && imgs[0].naturalWidth && imgs[1].naturalWidth);

		const draw = (force = false) => {
			if (!canDraw()) return;

			const layoutKey = getLayoutKey();
			if (!force && layoutKey === lastLayoutKey) return;

			lastLayoutKey = layoutKey;
			drawCorrespondences(card, imgs[0], imgs[1], svg, data);
		};

		const scheduleDraw = (force = false) => {
			if (!isVisible) return;
			if (resizeTimeoutId) clearTimeout(resizeTimeoutId);

			resizeTimeoutId = window.setTimeout(() => {
				resizeTimeoutId = 0;

				if (rafId) cancelAnimationFrame(rafId);
				rafId = requestAnimationFrame(() => {
					rafId = 0;
					draw(force);
				});
			}, force ? 0 : 120);
		};

		const showCorrespondences = () => {
			isVisible = true;

			toggle.textContent = tLabel('examples.pairs.card.hidecorrespondences');
			setOptionTitle(toggle, 'examples.pairs.card.hidecorrespondences');
			toggle.setAttribute('aria-pressed', 'true');

			svg.hidden = false;
			svg.removeAttribute('hidden');
			svg.style.display = 'block';

			lastLayoutKey = '';
			scheduleDraw(true);
		};

		const hideCorrespondences = () => {
			isVisible = false;

			toggle.textContent = tLabel('examples.pairs.card.showcorrespondences');
			setOptionTitle(toggle, 'examples.pairs.card.showcorrespondences');
			toggle.setAttribute('aria-pressed', 'false');

			svg.innerHTML = '';
			svg.hidden = true;
			svg.setAttribute('hidden', '');
			svg.style.display = 'none';

			lastLayoutKey = '';
		};

		const onToggleClick = () => {
			if (isVisible) hideCorrespondences();
			else showCorrespondences();
		};

		const onImageLoad = () => { scheduleDraw(true); };
		const onWindowResize = () => { scheduleDraw(false); };
		const onOrientationChange = () => { scheduleDraw(true); };
		toggle.addEventListener('click', onToggleClick);
		imgs.forEach((img) => { img.addEventListener('load', onImageLoad); });

		let resizeObserver = null;

		if ('ResizeObserver' in window) {
			resizeObserver = new ResizeObserver(() => { scheduleDraw(false); });
			resizeObserver.observe(card);
		}
		else {
			window.addEventListener('resize', onWindowResize);
			window.addEventListener('orientationchange', onOrientationChange);
		}

		card.__corrCleanup = () => {
			if (rafId) cancelAnimationFrame(rafId);
			if (resizeTimeoutId) clearTimeout(resizeTimeoutId);

			toggle.removeEventListener('click', onToggleClick);
			imgs.forEach((img) => { img.removeEventListener('load', onImageLoad); });

			if (resizeObserver) resizeObserver.disconnect();
			else {
				window.removeEventListener('resize', onWindowResize);
				window.removeEventListener('orientationchange', onOrientationChange);
			}

			delete card.__corrCleanup;
		};

		showCorrespondences();
	});
}

/** Draw inlier, outlier, and unused correspondence overlays for one RGB pair card. */
function drawCorrespondences(container, img1, img2, svg, data) {
	const containerRect = container.getBoundingClientRect();
	const rect1 = img1.getBoundingClientRect();
	const rect2 = img2.getBoundingClientRect();

	if (!rect1.width || !rect2.width || !img1.naturalWidth || !img2.naturalWidth) return;

	const total = Math.min(data.kp1.length, data.kp2.length);
	const unusedSet = new Set(Array.isArray(data.unusedKps) ? data.unusedKps.map(Number) : []);
	const inlierSet = buildInlierSet(total, data.inliers, data.unusedKps);

	const inlierIdx = [];
	const outlierIdx = [];
	const unusedIdx = [];

	for (let i = 0; i < total; i += 1) {
		if (unusedSet.has(i)) unusedIdx.push(i);
		else if (inlierSet.has(i)) inlierIdx.push(i);
		else outlierIdx.push(i);
	}

	const linesPerGroup = 120/(unusedIdx.length + inlierIdx.length + outlierIdx.length);

	svg.setAttribute('viewBox', `0 0 ${Math.round(containerRect.width)} ${Math.round(containerRect.height)}`);
	svg.innerHTML = '';

	drawKeypointGroup(svg, evenlySample(inlierIdx, inlierIdx.length*linesPerGroup), 'inlier', data, img1, img2, rect1, rect2, containerRect, true);
	drawKeypointGroup(svg, evenlySample(outlierIdx, outlierIdx.length*linesPerGroup), 'outlier', data, img1, img2, rect1, rect2, containerRect, true);
	drawKeypointGroup(svg, evenlySample(unusedIdx, unusedIdx.length*linesPerGroup), 'unused', data, img1, img2, rect1, rect2, containerRect, false);
}

/** Draw a sampled group of keypoints, optionally including correspondence lines. */
function drawKeypointGroup(svg, indices, kind, data, img1, img2, rect1, rect2, containerRect, includeLine) {
	indices.forEach((index) => {
		const [x1, y1] = projectPoint(data.kp1[index], rect1, containerRect, img1);
		const [x2, y2] = projectPoint(data.kp2[index], rect2, containerRect, img2);

		if (includeLine) appendLine(svg, x1, y1, x2, y2, kind);
		appendCircle(svg, x1, y1, kind);
		appendCircle(svg, x2, y2, kind);
	});
}

/** Add one SVG correspondence line. */
function appendLine(svg, x1, y1, x2, y2, kind) {
	const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
	line.setAttribute('x1', x1);
	line.setAttribute('y1', y1);
	line.setAttribute('x2', x2);
	line.setAttribute('y2', y2);
	line.setAttribute('class', `corr-line ${kind}`);
	svg.appendChild(line);
}

/** Add one SVG keypoint circle. */
function appendCircle(svg, x, y, kind) {
	const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
	circle.setAttribute('cx', x);
	circle.setAttribute('cy', y);
	circle.setAttribute('r', kind === 'unused' ? 0.8 : 1.1);
	circle.setAttribute('class', `corr-point ${kind}`);
	svg.appendChild(circle);
}

/** Project one keypoint from image coordinates into the overlay coordinate system. */
function projectPoint(point, imageRect, containerRect, imageEl) {
	const [xRaw, yRaw] = point || [0, 0];
	const x = imageRect.left - containerRect.left + (Number(xRaw) / imageEl.naturalWidth) * imageRect.width;
	const y = imageRect.top - containerRect.top + (Number(yRaw) / imageEl.naturalHeight) * imageRect.height;
	return [x.toFixed(2), y.toFixed(2)];
}

/** Evenly sample a long list of indices to keep overlays readable. */
function evenlySample(items, maxCount) {
	if (items.length <= maxCount) return items;

	const out = [];
	const step = items.length / maxCount;

	for (let i = 0; i < maxCount; i += 1) {
		out.push(items[Math.floor(i * step)]);
	}

	return out;
}


/** Normalize inlier representation into a set of active keypoint indices. */
export function buildInlierSet(total, inliers, unusedKps = []) {
	if (!Array.isArray(inliers)) return new Set();

	const unusedSet = new Set(Array.isArray(unusedKps) ? unusedKps.map((value) => Number(value)).filter((value) => Number.isInteger(value) && value >= 0 && value < total) : []);
	const isBinaryMask = inliers.every((value) => (typeof value === 'boolean' || value === 0 || value === 1));

	//console.log('Building inlier set with', { total, inliersLength: inliers.length, unusedKpsLength: unusedSet.size, inliers: inliers.filter(v => v == true).length });

	// Case 1: inliers is a full mask over all kp1/kp2. Example: kp1.length = 733, inliers.length = 733
	if (isBinaryMask && inliers.length === total) {
		const out = new Set();

		inliers.forEach((value, index) => { if (Boolean(value) && !unusedSet.has(index)) out.add(index); });
		return out;
	}

	// Case 2: inliers is a compact mask over only used keypoints. Example: kp1.length = 733, unused_kps.length = 2, inliers.length = 731
	if (isBinaryMask && inliers.length === total - unusedSet.size) {
		const out = new Set();
		let compactIndex = 0;

		for (let originalIndex = 0; originalIndex < total; originalIndex += 1) {
			if (unusedSet.has(originalIndex)) continue;
			if (Boolean(inliers[compactIndex])) out.add(originalIndex);
			compactIndex += 1;
		}

		return out;
	}

	console.warn('Inliers mask length does not match kp count.', { total, inliersLength: inliers.length, unusedKpsLength: unusedSet.size });
	return new Set();
}