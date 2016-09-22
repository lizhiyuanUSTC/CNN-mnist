function net = cnnbp(net, y)
% ���򴫲�
% e��¼��ǰ��ġ��в
    L = numel(net.layers);
    n = size(y, 2);   % ͼƬ��Ŀ
    net.layers{L}.e = (net.output - y) .* (net.output .* (1 - net.output)) / n;   % �����в�
    % ���������������ݶȣ������ڴ���ʱ�ı��������Ϊ���в��ǰ����ʱ��Ҫ��ǰ�Ĳ���
    net.layers{L}.dw = net.layers{L}.e * net.layers{L-1}.a';
    net.layers{L}.db = sum(net.layers{L}.e, 2);
    for l = L-1 : -1 : 2    
        if(strcmp(net.layers{l}.type, 'FullConnected'))
            net.layers{l}.e = net.layers{l+1}.w' * net.layers{l+1}.e .* (1 - net.layers{l}.a) .* net.layers{l}.a;
            net.layers{l}.dw = net.layers{l}.e * net.layers{l-1}.a' / n;
            net.layers{l}.db = sum(net.layers{l}.e, 2) / n;
        end

        if(strcmp(net.layers{l}.type, 'Flatten'))
            % Flatten��Ĳв�ֱ�ӵ�����һ��Ĳв�
            net.layers{l}.e = net.layers{l+1}.w' * net.layers{l+1}.e;
            e = net.layers{l}.e;
            if(strcmp(net.layers{l-1}.type, 'Convolution'))
                % ���Flatten���ǰһ���Ǿ���㣬�в���ǰ����ʱ��Ҫ��ǰһ�εĵ��������
                % (Ĭ��pooling��û�м����)
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


   
                

        
            
        