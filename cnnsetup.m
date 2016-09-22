function net = cnnsetup(net)
    assert(strcmp(net.layers{1}.type, 'Input'),'请确保您的第一层网络type为Input');
    net.layers{1}.mapsize = net.layers{1}.shape;   % 图片的高,宽,通道数
    fprintf(['Input layers: ' num2str(net.layers{1}.mapsize), '\n']);
    for l = 2: numel(net.layers)
        if strcmp(net.layers{l}.type, 'Pooling') 
            kernel_size = net.layers{l}.kernel_size;
            net.layers{l}.mapsize = zeros(1, 3);
            net.layers{l}.mapsize(1) = net.layers{l-1}.mapsize(1) / kernel_size;
            net.layers{l}.mapsize(2) = net.layers{l-1}.mapsize(2) / kernel_size;
            net.layers{l}.mapsize(3) = net.layers{l-1}.mapsize(3);
            assert(all(floor(net.layers{l}.mapsize)==net.layers{l}.mapsize), ['经下采样后图片宽和高必须是整数. 实际值: ' num2str(net.layers{l}.mapsize)]);
            fprintf(['Pooling layers: ' num2str(net.layers{l}.mapsize), '\n']);
        end
        if strcmp(net.layers{l}.type, 'Convolution')
            net.layers{l}.mapsize = zeros(1, 3);
            net.layers{l}.mapsize(1) = net.layers{l-1}.mapsize(1) - net.layers{l}.kernel_size + 1;
            net.layers{l}.mapsize(2) = net.layers{l-1}.mapsize(2) - net.layers{l}.kernel_size + 1;
            net.layers{l}.mapsize(3) = net.layers{l}.num_output;
            fprintf(['Convolution layers: ' num2str(net.layers{l}.mapsize), '\n']);
            inputmaps = net.layers{l-1}.mapsize(3);
            outputmaps = net.layers{l}.mapsize(3);
            fan = (inputmaps + outputmaps) * net.layers{l}.kernel_size ^ 2;
            for j = 1 : outputmaps
                for k = 1 : inputmaps
                    net.layers{l}.w{j}{k} = rand(net.layers{l}.kernel_size);
                    net.layers{l}.w{j}{k} = (net.layers{l}.w{j}{k} - 0.5) * 2 * sqrt(6 / fan);
                end
                net.layers{l}.b{j} = 0;
            end
        end
        
        if strcmp(net.layers{l}.type, 'Flatten')
           net.layers{l}.num_output = net.layers{l-1}.mapsize(1) * net.layers{l-1}.mapsize(2) * net.layers{l-1}.mapsize(3);
           fprintf(['Flatten layers: ' num2str(net.layers{l}.num_output), '\n']);
        end
        if strcmp(net.layers{l}.type, 'FullConnected')
           onum = net.layers{l}.num_output;
           fvnum = net.layers{l-1}.num_output;
           net.layers{l}.w = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
           net.layers{l}.b = zeros(net.layers{l}.num_output, 1);
           fprintf(['FullConnected layers: ' num2str(net.layers{l}.num_output), '\n']);
        end
    end
    



