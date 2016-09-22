function net = cnnapplygrad(net, learning_rate)
% ���²���
for l = 2 : numel(net.layers)
       if strcmp(net.layers{l}.type, 'Convolution')
           for c = 1 : net.layers{l}.mapsize(3)   % ���ͼƬͨ����
               for k = 1 : net.layers{l-1}.mapsize(3)   % ����ͼƬͨ����
                   net.layers{l}.w{c}{k} = net.layers{l}.w{c}{k} - learning_rate * net.layers{l}.dw{c}{k};
               end
               net.layers{l}.b{c} = net.layers{l}.b{c} -  learning_rate * net.layers{l}.db{c};
           end
       end
       
       if strcmp(net.layers{l}.type, 'FullConnected')
           net.layers{l}.w = net.layers{l}.w - learning_rate * net.layers{l}.dw;
           net.layers{l}.b = net.layers{l}.b - learning_rate * net.layers{l}.db;
       end
end