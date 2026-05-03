/* Dictionary defining the columns of the results table, including their keys, labels, visibility, and value formatting functions. */
const tableDict = (benchmarkMode) => [
	{
        key: 'group',
        label: 'Group',
        visible: false,
        formatValue: (value) => String(value)
    },
	{
        key: 'dataset',
        label: benchmarkMode === 'standard' ? 'Dataset' : 'Scene',
        visible: true,
        formatValue: (value) => String(value)
    },
	{
        key: 'iters',
        label: 'Iters',
        visible: true,
        formatValue: (value) => String(value)
    },
	{
        key: 'mde',
        label: 'MDE',
        visible: true,
        formatValue: (value) => String(value)
    },
	{
        key: 'solver',
        label: 'Solver',
        visible: true,
        formatValue: (value) => String(value)
    },
	{
        key: 'pose_mAA_10',
        label: 'mAA(10°)',
        visible: true,
        formatValue: (value) => typeof value === 'number' ? value.toFixed(2) : String(value)
    },
	{
        key: 'mean_mde_runtime',
        label: 'MDE Runtime (ms)',
        visible: true,
        formatValue: (value) => typeof value === 'number' ? value.toFixed(2) : String(value)
    },
	{
        key: 'mean_inliers',
        label: 'Inliers (%)',
        visible: true,
        formatValue: (value) => typeof value === 'number' ? value.toFixed(2) : String(value)
    }
];

/* Create a Map from the table dictionary for easy access to column definitions by their keys. */
export const tableDictByKey = (benchmarkMode) => new Map(tableDict(benchmarkMode).map((column) => [column.key, column]));