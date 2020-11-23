import numpy as np

NOISE_TYPES = ['normal', 'rain', 'fog', 'snow', 'night']

def add_noise(x, noise_type='normal'):
	assert noise_type in NOISE_TYPES 
	if noise_type == 'normal':
		return x
	elif noise_type == 'rain':
		return x + 1
	elif noise_type == 'fog':
		return x + 2
	elif noise_type == 'snow':
		return x + 3
	elif noise_type == 'night':
		return x + 4


def get_minibatch(X, y, batch_size=32, shuffle=True, \
	              drop_last=True, augementing=True):
	idx = np.arange(len(X))
	if drop_last:
		n_batches = len(idx) // batch_size
	else:
		n_batches = np.ceil(len(idx) / batch_size).astype(np.int32)
	if shuffle:
		np.random.shuffle(idx)

	for b in range(n_batches):
		left_idx  = b*batch_size
		right_idx = min( (b+1)*batch_size, len(idx))
		batch_idx = idx[left_idx:right_idx]
		X_batch, y_batch = X[batch_idx], y[batch_idx]
		if augementing:
			global_idx = 0
			aug_bs = batch_size // len(NOISE_TYPES)
			for i,noise in enumerate(NOISE_TYPES):
				for x in X_batch[i*aug_bs: (i+1)*aug_bs]:
					X_batch[global_idx] = add_noise(x, noise)
					global_idx += 1
		yield X_batch, y_batch


if __name__ == '__main__':
	X = np.random.rand(36000, 49, 49, 3)
	y = np.random.randint(0,103,size=len(X))
	model = KERASMODEL

	history_loss_valid = []
	for epoch in range(100):
		for xb, yb in get_minibatch(X, y, batch_size=32):
			loss_train = model.train_on_batch(xb, yb)
		
		loss_valid = 0.0
		for xb, yb in get_minibatch(X_valid, y_valid, batch_size=32,
									shuffle=False, augementing=False, drop_last=False):
			loss_valid += len(xb)*(-np.log(model.predict(xb)[np.arange(len(yb)),yb]))
		loss_valid = loss_valid / len(X_valid)
		history_loss_valid.append(loss_valid)









