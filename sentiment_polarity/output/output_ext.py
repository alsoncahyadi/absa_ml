import dill

n = [
	'mean_test_f1_macro',
	'mean_test_precision_macro',
	'mean_test_recall_macro',
	'param_activation',
	'param_batch_size',
	'param_epochs',
	'param_conv_activation',
	'param_conv_l2_regularizer',
	'param_dense_activation',
	'param_dense_l2_regularizer',
	'param_dropout_rate',
	'param_filters',
	'param_kernel_size',
	'param_loss_function',
	'param_optimizer',
	'param_units'
]

cats = [
	'food', 'place', 'service', 'price'
]

for cat in cats:
	with open('gridsearch_cnn_{}.pkl'.format(cat), 'rb') as fi:
		g = dill.load(fi)

	print(g.keys())

	l = len(g['mean_test_f1_macro'])

	with open('gridsearch_cnn_{}.csv'.format(cat), 'w') as fo:
		fo.write('	'.join(n) + '\n')
	for i in range(l):
		out = [str(g[k][i]) for k in n]
		out = '	'.join(out) + '\n'
		with open('gridsearch_cnn_{}.csv'.format(cat), 'a') as fo:
			fo.write(out)