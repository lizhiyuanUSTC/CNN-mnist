% ����ʾ�������ڽ����ݼ���test_x�����ѡȡ15��ͼƬ������

clear all;
load mnist_uint8;
load net;
row = 10; % ÿ����ʾ��ͼƬ��Ŀ
col = 10; % ÿ����ʾ��ͼƬ��Ŀ
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
        title(['ʵ��ֵ:' num2str(y(k)) ',Ԥ��ֵ:' num2str(label(k))]);
    end
end