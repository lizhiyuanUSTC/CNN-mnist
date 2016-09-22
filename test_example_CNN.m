
load mnist_uint8
train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');
net.layers = {
    struct('type', 'Input', 'shape', [28, 28, 1])
    struct('type', 'Convolution', 'kernel_size', 5, 'num_output', 6)
    struct('type', 'Pooling', 'kernel_size', 2)
    struct('type', 'Convolution', 'kernel_size', 5, 'num_output', 12)
    struct('type', 'Pooling', 'kernel_size', 2)
    struct('type', 'Flatten')
    struct('type', 'FullConnected', 'num_output', 10)
};


opts.alpha = 1;    % 学习率
opts.batchsize = 50;  % 训练集batchsize
opts.numepochs = 10;  % 便利整个训练集10次
opts.val_loop = 100;  % 每隔100次进行一次测试
net = cnnsetup(net);
net = cnntrain(net, train_x, train_y, test_x, test_y, opts);
save('net.mat', 'net');
