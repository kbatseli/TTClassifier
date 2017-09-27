%% Experiments for MNIST
%% Train the TT cores using one-to-all decision strategy
load Data_mnist
d=35;   n=1;   r=8;   gamma=1e-2;
p=(d-20)/5+1;    % FEATURE save the pretrained CNN feature for d = 20:5:40   
fimgs=feature{p}(1:60000, :);    ftimgs=feature{p}(60001:end,:);

x=cell(10,1);   res=cell(10,1);   err=cell(10,1);   
for i=1:10
    tic;
    %[x{i}, res{i}, err{i}]=ttls(fimgs, rflabels(:,i), n, r, gamma);
    [x{i}, res{i}, err{i}]=ttlr(fimgs, rflabels(:,i), n, r, gamma);
    toc; 
end
%save MNIST_TTLS x res err
%save MNIST_TTLR x res err

%% Check the error rate of training data %%
score=zeros(60000, 10);
for i=1:10
    score(:,i)=check(fimgs, x{i});
end
[~, ind]=max(score,[],2);
ind=ind-1;
train_err=sum(ind ~= labels)/60000

%% Check the error rate of test data
score=zeros(10000,10);
for i=1:10
    score(:,i)=check(ftimgs, x{i});
end
[~, ind]=max(score,[],2);
ind=ind-1;
test_err=sum(ind ~= tlabels)/10000

