function test_example_CNN
load mnist_uint8;


train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network 
%will run 1 epoch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};


opts.alpha = 1;             %2-->90.34; 1-->88.87; 0.8-->87.65; 0.5-->84.47
opts.batchsize = 50;        %50-->88.87; 40-->90.01; 30-->91.4; 20-->93.70
opts.numepochs = 1;        %1-->88.87; 5-->95.44; 10-->97.27
%tic
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);
%toc
[er, bad] = cnntest(cnn, test_x, test_y);
Performa=(1-er)*100
%plot mean squared error
%save cnnHS_0115it50b;   %data untuk save
%save newdatacnn10.mat;
%save ('data.mat', cnn); 
figure; plot(cnn.rL);
assert(er<0.12, 'Too big error');
