/** Available CSV sources for the standard benchmark, keyed by source identifier. */
export const STANDARD_TABLE_SOURCES = {
	loma: {
		key: 'controls.select.source.loma',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/standard_loma_slim_pose_results.csv'
	},
	splg: {
		key: 'controls.select.source.splg',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/standard_splg_slim_pose_results.csv'
	}
};

/** Available CSV sources for the group-based (d2p) benchmark, keyed by source identifier. */
export const GROUP_TABLE_SOURCES = {
	loma: {
		key: 'controls.select.source.loma',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/d2p_loma_slim_pose_results.csv'
	},
	splg: {
		key: 'controls.select.source.splg',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/d2p_splg_slim_pose_results.csv'
	}
};

/** URL to the source folder for image examples. */
export const IMG_EXAMPLES_SOURCE = 'https://davinci.fmph.uniba.sk/~kocur15/d2p_examples/';