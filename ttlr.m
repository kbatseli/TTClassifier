function [x, res, err]=ttlr(a, b, n, r, gamma, varargin)
% [x, res, err]=ttlr(a, b, n, r, gamma) or [x, res, err]=ttlr(a, b, n, r, gamma, x0)
% --------------------------------------------------------------------------------------------
% Tensor Train polynomial classifier trained with logistic regression and
% regularization. 
%
% x         =   cell, x{i} contains the ith Tensor Train core of the
% 				polynomial classifier,
%
% res		=	vector, res(i) contains the logistic regression loss
%				function value at iteration i,
% 
% err       =   vector, e(i) contains the sign error rate at iteration i,
%
% a         =   matrix, N x d matrix containing N samples of d-dimensional features,
%
% b         =   matrix, N x 1 matrix that contains N class labels for each of the features,
%
% n			=	scalar or vector, if scalar then n is the largest degree
%				of all features, otherwise n(i) contains the largest degree
%				of feature i,
%
% r         =   scalar or vector, vector contains the TT ranks r_1 up to r_{d-1}, since
%               r_0=r_d=1. If a scalar, then uniform TT ranks are chosen,
%
% gamma		=   scalar, regularization parameter,
%
% x0 		=	cell, optional initial TT classifier. Default: {}, if an
%				initial TT classifier is used, then the TT ranks of the
%				initial classifier will be used.    
%
% Reference
% ---------
%
% Parallelized Tensor Train Learning of Polynomial Classifiers
%
% https://arxiv.org/abs/1612.06505
% 
% 2016, Zhongming Chen

% Regularized TTLR with L2-norm
[N, d]=size(a);    
if numel(n)==1 
   n= n*ones(1,d);    % the largest degree for each variable
end
if numel(r)==1
  r=r*ones(1,d+1); 
  r(1) = 1;    r(d+1)=1;
end

% Initialize tensor train X and matrix group Md
if ~isempty(varargin)
    x=varargin{1};
    for i=1:d
        r(i)=size(x{i},1);
    end
else
    x=cell(1,d);
    for i=1:d
        x{i}=randn(r(i),n(i)+1,r(i+1));
        x{i}=x{i}/norm(x{i}(:));
    end
end

Matd=cell(1,d+1);
Matd{d+1}=ones(N,r(d+1));    R=1;
for i=1:d
    temp=R*reshape(x{i}, r(i), (n(i)+1)*r(i+1));
    r(i)=size(R,1);
    temp=reshape(temp, r(i)*(n(i)+1), r(i+1));
    [Q, R]=qr(temp, 0);
    x{i}=reshape(Q, r(i), n(i)+1, size(Q,2));
end
Matn=cell(1,d+1);
Matn{1}=1;    Matn{d+1}=1;
for i=d:-1:2
    temp=reshape(x{i}, r(i)*(n(i)+1),r(i+1))*R;
    r(i+1)=size(R,2);
    temp=reshape(temp, r(i), (n(i)+1)*r(i+1));
    [Q,R]=qr(temp',0);    Q=Q';    R=R';
    x{i}=reshape(Q, size(Q,1), n(i)+1, r(i+1));
    Mati=repmat(a(:,i), 1, n(i)+1).^(kron(0:n(i), ones(N,1)));
    Matd{i}=dotkron(Mati, Matd{i+1})*Q';
    
    temp=reshape(permute(x{i}, [2 1 3]), n(i)+1, size(Q,1)*r(i+1));
    temp=sum(dotkron(temp,temp));
    temp=permute(reshape(temp, size(Q,1), r(i+1),...
        size(Q,1), r(i+1)), [1 3 2 4]);
    Matn{i}=reshape(temp, size(Q,1)^2, r(i+1)^2)*Matn{i+1};
end
temp=reshape(x{1}, r(1)*(n(1)+1), r(2))*R;
r(2)=size(R,2);
x{1}=reshape(temp, r(1), n(1)+1, r(2));
Matd{1}=ones(N,r(1));


ite=0;    itemax=(d-1)*4;    gap=inf;     
while  ite < itemax  
    ite=ite+1;    loopind=mod(ite-1,2*(d-1))+1;
    dir=(loopind < d-0.5);
    if loopind <= d
        ind=loopind;
    else
        ind=2*d-loopind;
    end
    
    Mati=repmat(a(:,ind), 1, n(ind)+1).^(kron(0:n(ind), ones(N,1)));
    C=dotkron(Matd{ind}, Mati, Matd{ind+1});
    vn=reshape(kron(reshape(Matn{ind+1}, r(ind+1), r(ind+1)),...
        reshape(Matn{ind}, r(ind), r(ind))), r(ind)*r(ind+1)*r(ind)*r(ind+1),1);

    [y, fval, normA]=my_newton3_plus(C, b, gamma, x{ind}, vn);  %gamma 
    
    res(ite)=fval;                       % LR loss function
    err(ite)=sum(sign(C*y) ~= b)/N;      % Sign error rate
    
    if dir==1 
        yy=reshape(y, r(ind)*(n(ind)+1), r(ind+1));
        [Q,R]=qr(yy,0);    
        x{ind+1}=R*reshape(x{ind+1}, r(ind+1), (n(ind+1)+1)*r(ind+2));
        r(ind+1)=size(Q,2);
        x{ind+1}=reshape(x{ind+1}, r(ind+1), n(ind+1)+1, r(ind+2));
        x{ind}=reshape(Q, r(ind), n(ind)+1, r(ind+1));
        Matd{ind+1}=dotkron(Matd{ind}, Mati)*Q;       
        
        temp=reshape(permute(x{ind}, [2 1 3]), n(ind)+1, r(ind)*size(Q,2));
        temp=sum(dotkron(temp,temp));
        temp=permute(reshape(temp, r(ind), size(Q,2),...
            r(ind), size(Q,2)), [1 3 2 4]);
        Matn{ind+1}=Matn{ind}*reshape(temp, r(ind)^2, size(Q,2)^2);
    end
    if dir==0
        yy=reshape(y, r(ind),(n(ind)+1)*r(ind+1));
        [Q,R]=qr(yy',0);    Q=Q';    R=R';
        x{ind-1}=reshape(x{ind-1}, r(ind-1)*(n(ind-1)+1), r(ind))*R;
        r(ind)=size(Q,1);
        x{ind-1}=reshape(x{ind-1}, r(ind-1), n(ind-1)+1, r(ind));
        x{ind}=reshape(Q, r(ind), n(ind)+1, r(ind+1));
        Matd{ind}=dotkron(Mati, Matd{ind+1})*Q';
        
        temp=reshape(permute(x{ind}, [2 1 3]), n(ind)+1, size(Q,1)*r(ind+1));
        temp=sum(dotkron(temp,temp));
        temp=permute(reshape(temp, size(Q,1), r(ind+1),...
            size(Q,1), r(ind+1)), [1 3 2 4]);
        Matn{ind}=reshape(temp, size(Q,1)^2, r(ind+1)^2)*Matn{ind+1};
    end
end







end
