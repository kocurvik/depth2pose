export const tableDict = [
	{
        key: 'dataset',
        label: 'Dataset',
        formatValue: (value) => String(value)
    },
	{
        key: 'iters',
        label: 'Iters',
        formatValue: (value) => String(value)
    },
	{
        key: 'mde',
        label: 'MDE',
        formatValue: (value) => String(value)
    },
	{
        key: 'solver',
        label: 'Solver',
        formatValue: (value) => String(value)
    },
	{
        key: 'pose_mAA_10',
        label: 'mAA(10°)',
        formatValue: (value) => typeof value === 'number' ? value.toFixed(2) : String(value)
    },
	{
        key: 'mean_mde_runtime',
        label: 'MDE Runtime (ms)',
        formatValue: (value) => typeof value === 'number' ? value.toFixed(2) : String(value)
    },
	{
        key: 'mean_inliers',
        label: 'Inliers (%)',
        formatValue: (value) => typeof value === 'number' ? (value * 100).toFixed(2) : String(value)
    }
];