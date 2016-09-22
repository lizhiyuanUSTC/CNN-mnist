% 本演示程序用于将数据集中test_x中随机选取15张图片做测试

clear all;
load mnist_uint8;
load net;
row = 10; % 每列显示的图片数目
col = 10; % 每行显示的图片数目
test_x = double(reshape(test_x',28,28,10000))/255;
test_y = double(test_y');
index = randi([1 10000], row * col, 1);
x = test_x(:, :, index);
y = test_y(:, index);
label = cnnpredict(net, x);
[~, y] = max(y);
y = y - 1;
for i = 1 : row
    for j = 1 : col
        k = (i-1)*col+j;
        img = x(:, :, k)';
        subplot(row, col, k);
        imshow(img);
        title(['实际值:' num2str(y(k)) ',预测值:' num2str(label(k))]);
    end
end