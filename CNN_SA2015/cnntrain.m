function net = cnntrain(net, x, y, opts)
    m = size(x, 3);                     %Jumlah data training: 60.000
    numbatches = m / opts.batchsize;    %Jumlah batch: 1200
    
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    tic;
    
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)])
        
        kk = randperm(m);
        for l = 1 : numbatches
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            
            %%net = SAoptimasiCNNmay2015(net, batch_y);
            %net = SAoptimasiCNN(net, batch_y);
            %net = DEoptimasiCNN(net, batch_y); 
            %%net = SAoptimasiCNNbaru(net, batch_y);
            net= HSoptimasiCNNnew(net, batch_y);
            %net= HSoptimasiCNN005(net, batch_y);
            %%net = SAoptimasiCNNbaru01(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            iterasi=l
        end
        
        
    end
    toc;
     %net = SAoptimasiCNN(net, batch_y);
end
