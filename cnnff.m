function net = cnnff(net, x, varargin)
% ����ǰ�򴫲�
% x ��һ��ά����,h*w*n*c
% h ��ͼƬ�߶�
% w ��ͼƬ���
% n ��ͼƬ��Ŀ
% c ��ͨ����
    n = size(x, 3);
    c = size(x, 4);   % ͼƬͨ����Ŀ
    for k = 1 : c
        net.layers{1}.a{c} = x(:, :, :, c);
    end

    for l = 2 : numel(net.layers)
       if strcmp(net.layers{l}.type, 'Convolution')
           for c = 1 : net.layers{l}.mapsize(3)   % ���ͼƬͨ����
               z = zeros(net.layers{l}.mapsize(1), net.layers{l}.mapsize(2), n);
               for k = 1 : net.layers{l-1}.mapsize(3)   % ����ͼƬͨ����
                   z = z + convn(net.layers{l-1}.a{k}, net.layers{l}.w{c}{k}, 'valid');
               end
               net.layers{l}.a{c} = sigmoid(z + net.layers{l}.b{c});
           end
       end

       if strcmp(net.layers{l}.type, 'Pooling')
           kernel = ones(net.layers{l}.kernel_size);
           for c = 1 : net.layers{l}.mapsize(3)   % ���ͼƬͨ����
               % ����mean pooling
               z = convn(net.layers{l-1}.a{c}, kernel, 'valid') / (net.layers{l}.kernel_size^2);
               net.layers{l}.a{c} = z(1:net.layers{l}.kernel_size:end, 1:net.layers{l}.kernel_size:end, :) ;
           end
       end
       
       if strcmp(net.layers{l}.type, 'Flatten')
           % ����һ�����������ͼ�ϲ���һ��������
           net.layers{l}.a = [];
           sa = net.layers{l-1}.mapsize(1:2);
           for c = 1 : net.layers{l-1}.mapsize(3)   % ���ͼƬͨ����
               z = reshape(net.layers{l-1}.a{c}, sa(1) * sa(2), n);
               net.layers{l}.a = [net.layers{l}.a; z];
           end    
       end
       
       if strcmp(net.layers{l}.type, 'FullConnected')
           % ȫ���Ӳ㣬����sigmoid�����������ÿ�����ĸ���
           dim = net.layers{l-1}.num_output;
           a = net.layers{l-1}.a;
           w = net.layers{l}.w;
           a(dim + 1, :) = 1;
           w(:, dim + 1) = net.layers{l}.b;
           net.layers{l}.a = sigmoid(w * a);
           net.output = net.layers{l}.a;
       end
    end
    
    if nargin == 3
        y = varargin{1};
        net.loss = 1/2 * sum(sum((y - net.output).^2)) / n;
        [~, predict_label] = max(net.output);   % ����Ԥ�����
        [~, correct_label] = max(y);   % ʵ�����
        net.accuracy = sum(predict_label == correct_label) / n;   % Ԥ��׼ȷ��
    end
end