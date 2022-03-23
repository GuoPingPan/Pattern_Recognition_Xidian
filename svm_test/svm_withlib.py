from libsvm.svmutil import *
# y,x = svm_read_problem('./data/iris.train')
''' load training data '''
y,x = svm_read_problem('./data/sonar.train')
prob = svm_problem(y,x)
# param = svm_parameter('-t 0 -c 512.0 -g 0.0078125 ')
# param = svm_parameter('-t 1 -c 512.0 -g 0.5 ')
# param = svm_parameter('-t 2 -c 32.0 -g 0.0078125')
# param = svm_parameter('-t 3 -c 2048.0 -g 0.007')

''' load the best parameters '''
param = svm_parameter('-t 2 -c 32.0 -g 0.03125')

''' train '''
model = svm_train(prob,param)

''' save model '''
# svm_save_model('./iris.model',model)

''' load model '''
# svm_load_model('./iris.model',model)


# @params p_label:预测标签
# @params p_acc:(预测精确度，均值，回归平方相关系数)
# @params p_val:在指定参数'-b 1'时返回判定系数

print('Train:')
p_label,p_acc,p_val = svm_predict(y,x,model)
print(f'Mean:{p_acc[1]}')
print(f'回归平方相关系数:{p_acc[2]}')

print('Test:')
# yt, xt = svm_read_problem('./data/iris.test')  # 读取测试集的数据
yt, xt = svm_read_problem('./data/sonar.test')
p_label, p_acc, p_val = svm_predict(yt, xt, model)
print(f'Mean:{p_acc[1]}')
print(f'回归平方相关系数:{p_acc[2]}')
