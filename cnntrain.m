function net = cnntrain(net, x, y, test_x, test_y, opts)
num_images = size(x, 3);
num_batch = floor(num_images / opts.batchsize);
net.train_loss_history = zeros(1, opts.numepochs * num_batch);
net.train_accu_history = zeros(1, opts.numepochs * num_batch);
net.val_loss_history = zeros(1, floor(opts.numepochs * num_batch / opts.val_loop));
net.val_accu_history = zeros(1, floor(opts.numepochs * num_batch / opts.val_loop));
s = 1;
for k = 1:opts.numepochs
    for l = 1:num_batch
        y_batch = y(:, (l-1)*opts.batchsize+1 : l*opts.batchsize);
        if size(x, 4) > 1
            x_batch = x(:, :, (l-1)*opts.batchsize+1 : l*opts.batchsize, :);
        else
            x_batch = x(:, :, (l-1)*opts.batchsize+1 : l*opts.batchsize);
        end
        net = cnnff(net, x_batch, y_batch);
        net = cnnbp(net, y_batch);
        net = cnnapplygrad(net, opts.alpha);
        fprintf(['epoch ' num2str(k) ' batch ' num2str(l) '\n']);
        fprintf(['    train loss:' num2str(net.loss) ',train accuracy' num2str(net.accuracy) '\n']);
        net.train_loss_history((k-1)*num_batch+l) = net.loss;
        net.train_accu_history((k-1)*num_batch+l) = net.accuracy;
        if(mod(s, opts.val_loop) == 0)
            % 在验证集中随机采样200张图片进行验证
            index = randi([1, size(train_y, 3)], 200);
            val_x = test_x(:, :, index);
            val_y = test_y(:, index);
            val_net = cnnff(net, val_x, val_y);
            net.val_loss_history(s) = val_net.loss;
            net.val_accu_history(s) = val_net.accuracy;
            s = s + 1;
        end
    end
    
end