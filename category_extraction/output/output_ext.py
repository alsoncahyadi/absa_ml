import dill


with open('gridsearch_cnn.pkl', 'rb') as fi:
	g = dill.load(fi)

print(g.keys())

l = len(g['mean_test_f1_macro'])

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

with open('gridsearch_cnn.csv', 'w') as fo:
	fo.write('	'.join(n) + '\n')
for i in range(l):
	out = [str(g[k][i]) for k in n]
	out = '	'.join(out) + '\n'
	with open('gridsearch_cnn.csv', 'a') as fo:
		fo.write(out)