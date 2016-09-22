function net = cnnbp(net, y)
% 反向传播
% e记录当前层的“残差”
    L = numel(net.layers);
    n = size(y, 2);   % 图片数目
    net.layers{L}.e = (net.output - y) .* (net.output .* (1 - net.output)) / n;   % 输出层残差
    % 计算输出层参数的梯度，不能在传播时改变参数，因为“残差”向前传递时需要当前的参数
    net.layers{L}.dw = net.layers{L}.e * net.layers{L-1}.a';
    net.layers{L}.db = sum(net.layers{L}.e, 2);
    for l = L-1 : -1 : 2    
        if(strcmp(net.layers{l}.type, 'FullConnected'))
            net.layers{l}.e = net.layers{l+1}.w' * net.layers{l+1}.e .* (1 - net.layers{l}.a) .* net.layers{l}.a;
            net.layers{l}.dw = net.layers{l}.e * net.layers{l-1}.a' / n;
            net.layers{l}.db = sum(net.layers{l}.e, 2) / n;
        end

        if(strcmp(net.layers{l}.type, 'Flatten'))
            % Flatten层的残差直接等于下一层的残差
            net.layers{l}.e = net.layers{l+1}.w' * net.layers{l+1}.e;
            e = net.layers{l}.e;
            if(strcmp(net.layers{l-1}.type, 'Convolution'))
                % 如果Flatten层的前一层是卷积层，残差向前传递时需要和前一次的导数做点乘
                % (默认pooling层没有激活函数)
                 e = e .* net.layers{l}.a .* (1 - net.layers{l}.a);
            end
            h = net.layers{l-1}.mapsize(1);
            w = net.layers{l-1}.mapsize(2);
            c = net.layers{l-1}.mapsize(3);
            
            for k = 1 : c
                z = e((k-1)*h*w+1 :k*h*w, :);
                z = reshape(z, h, w, n);
                net.layers{l-1}.e{k} = z;
            end
            
        end
        
        if(strcmp(net.layers{l}.type, 'Pooling'))
            s = net.layers{l}.kernel_size;
            for k = 1 : net.layers{l}.mapsize(3)
                net.layers{l-1}.e{k} = expand(net.layers{l}.e{k}, [s, s, 1]) / s^2;
                net.layers{l-1}.e{k} = net.layers{l-1}.e{k} .* (net.layers{l-1}.a{k} .* (1 - net.layers{l-1}.a{k}));
            end     
        end
            
        if(strcmp(net.layers{l}.type, 'Convolution'))
            for c = 1 : net.layers{l-1}.mapsize(3)
                z = zeros(net.layers{l-1}.mapsize(1), net.layers{l-1}.mapsize(2), n);
                for k = 1 : net.layers{l}.mapsize(3)  
                    z = z + convn(net.layers{l}.e{k}, rot180(net.layers{l}.w{k}{c}), 'full');    
                end
                net.layers{l-1}.e{c} = z;
            end
            
                
            for k = 1 : net.layers{l}.mapsize(3) 
                net.layers{l}.db{k} = sum(net.layers{l}.e{k}(:));
                for c = 1 : net.layers{l-1}.mapsize(3)
                    net.layers{l}.dw{k}{c} = convn(flipall(net.layers{l-1}.a{c}), net.layers{l}.e{k}, 'valid');     
                end      
            end 
        end
    end
end


   
                

        
            
        