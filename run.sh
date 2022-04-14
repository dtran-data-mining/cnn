conda activate hw-env
python3 main.py -mode=train -num_epoches=50 -ckp_path=checkpoint \
-learning_rate=0.01 -dropout=0.5 -p_huris=0.8 -batch_size=100 \
-channel_out1=64 -channel_out2=64 -rotation=15 \
-fc_hidden1=500 -fc_hidden2=100 -MC=20
