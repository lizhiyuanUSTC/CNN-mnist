function label = cnnpredict(net, x)
    net = cnnff(net, x);
    [~, label] = max(net.output);
    label = label - 1;