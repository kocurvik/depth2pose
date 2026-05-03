/* Available CSV sources for the standard benchmark, keyed by source identifier. */
export const STANDARD_TABLE_SOURCES = {
	loma: {
		label: 'LoMa',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/standard_loma_slim_pose_results.csv'
	},
	splg: {
		label: 'SP+LG',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/standard_splg_slim_pose_results.csv'
	}
};

/* Available CSV sources for the group-based (d2p) benchmark, keyed by source identifier. */
export const GROUP_TABLE_SOURCES = {
	loma: {
		label: 'LoMa',
		url: 'https://raw.githubusercontent.com/kocurvik/depth2pose/main/csv_results/d2p_slim_pose_results.csv'
	}
};

/* BASE URL to the folder containing image examples. */
export const IMG_EXAMPLES_BASE_URL = 'https://raw.githubusercontent.com/lbujnak/depth2pose_webdata/main/benchmark/d2p_examples';

/* URL to the JSON file for listing the contents of the d2p_examples folder. */
export const IMG_EXAMPLES_CONTENTS_URL = 'https://api.github.com/repos/lbujnak/depth2pose_webdata/contents/benchmark/d2p_examples?ref=main';