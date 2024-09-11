def RUN(block):	
	import numpy as np
	import matplotlib.pyplot as plt
	from keras.models import Sequential
	from keras.layers import Dense,RNN
	from math import sqrt
	from sklearn.metrics import mean_squared_error
	from sklearn.metrics import mean_absolute_error


	def NetPlot(net_history):
		history=net_history.history
		losses=history['loss']
		val_losses=history['val_loss']
		plt.figure('Loss Diagram')
		plt.title('Loss of Training Trend')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.plot(losses)
		plt.plot(val_losses)
		plt.legend(['Training Data','Validation Data'])      
		plt.show()


	def split_sequence(sequence, n_steps_in, n_steps_out):
		X, y = list(), list()
		for i in range(len(sequence)):
			end_ix = i + n_steps_in
			out_end_ix = end_ix + n_steps_out
			if out_end_ix > len(sequence):
				break
			seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
			X.append(seq_x)
			y.append(seq_y)
		return np.array(X), np.array(y)


	path='./dataset/Countries/Australia.txt'
	data0=np.loadtxt(path)

	# item 0: New Cases (daily)
	# item 1: New Cases (cumulative)
	# item 2: New Deaths (daily)
	# item 3: New Deaths (cumulative)
	data=[]
	for item in data0:
		data.append(item[3])
	data=np.array(data)


	n_steps_in,n_steps_out=1,block

	x, y = split_sequence(data, n_steps_in, n_steps_out)

	x_train=x[:-100]
	x_test=x[-100:]
	y_train=y[:-100]
	y_test=y[-100:]

	lst_x=[]
	lst_y=[]
	for i in range(0,100,block):
		lst_x.append(x_test[i])
		lst_y.append(y_test[i])

	x_test=np.array(lst_x)
	y_test=np.array(lst_y)


	n_features = 1
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))


	model = Sequential()
	model.add(RNN(50,activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
	model.add(RNN(50, activation='relu', return_sequences=True))
	model.add(RNN(50,activation='relu'))
	model.add(Dense(n_steps_out))

	model.compile(optimizer='adam', loss='mse')
	net=model.fit(x_train, y_train, epochs=200,validation_split=0.2)


	predicteds=model.predict(x_test)
	actuals=y_test

	MSE=mean_squared_error(actuals,predicteds)
	RMSE=sqrt(mean_squared_error(actuals,predicteds))
	MAE=mean_absolute_error(actuals,predicteds)



	actuals_path=f'./results/actuals_{block}_days_GRU.txt'
	predicteds_path=f'./results/predicteds_{block}_days_GRU.txt'
	metrics_path=f'./results/metrics_{block}_days_GRU.txt'

	f1=open(actuals_path,'a')
	f2=open(predicteds_path,'a')
	f3=open(metrics_path,'a')
	for i in range(len(predicteds)):
		st1=str(actuals[i])
		st1=st1.replace(']','')
		st1=st1.replace('[','')
		f1.write(st1+'\n\n')

		st2=str(predicteds[i])
		st2=st2.replace(']','')
		st2=st2.replace('[','')
		f2.write(st2+'\n\n')

	f3.write('MSE: '+str(MSE)+'\nRMSE: '+str(RMSE)+'\nMAE: '+str(MAE))

	f1.close()
	f2.close()
	f3.close()

	print(MSE)

	actuals=actuals.flatten()
	predicteds=predicteds.flatten()


	plt.figure(f'Forecasting {block} days_plot_GRU',dpi=200)
	plt.plot(actuals,color='black')
	plt.plot(predicteds,color='green')
	plt.xlabel('Day')
	plt.ylabel('New Case Rate')
	plt.legend(['Actuals','Predicts'])
	plt.savefig(f'./results/Forecasting {block} days_plot_GRU')

	plt.figure(f'Forecasting {block} days_scatter_GRU',dpi=200)
	plt.plot(actuals,actuals,color='red')
	plt.plot(actuals,predicteds,'bo',color='green')
	plt.xlabel('Observation')
	plt.ylabel('Predict')
	plt.legend(['Actuals','Predicts'])
	plt.savefig(f'./results/Forecasting {block} days_scatter_GRU')


def RunALL():
	periods=[1,3,7,10,14]
	for i in periods:
		RUN(i)


RunALL()
