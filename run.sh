# macOS specific - replace path with your conda.sh path
source /Users/dtran/opt/anaconda3/etc/profile.d/conda.sh
# activate hw-env conda virtual environment
conda activate hw-env
# execute main python script
python3 main.py -mode=test -num_epoches=50 -ckp_path=checkpoint \
-learning_rate=0.005 -decay=0.075 -dropout=0.5 -batch_size=100 \
-channel_out1=64 -channel_out2=64 -rotation=15 \
-fc_hidden1=500 -fc_hidden2=100
