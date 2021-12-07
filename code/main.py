import classfier
from documents.L1W2.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_num=train_set_x_orig.shape[0]
test_num=test_set_x_orig.shape[0]
print("训练样本数："+str(train_num))
print("测试样本数："+str(test_num))


train_set_x_flattened=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flattened=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
dimension=train_set_x_flattened.shape[0]
print("单样本向量维度:"+str(dimension))

train_set_x=train_set_x_flattened/255
test_set_x=test_set_x_flattened/255

d= classfier.classfier(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 5000000, learning_rate = 0.005, print_cost=False)
print(d);
