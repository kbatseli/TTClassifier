function [x, fx, normA]=my_newton3_plus(C, b, gamma, x0, vn)
% Minimize the logistic regression cost function
%   -(1/N)*\sum_i [(1+b_i)/2*log(sigmoid(x'*c_i) + ...
%                 (1-b_i)/2*log(1-sigmoid(x'*c_i) ]
% Using NEWTON method, fast
% Two disadvantage: 
%    1. Computing the inverse of Hessian
%    2. Without truncation for Sigmoid, FX can be INF due to rounding error
% Note that C and X0 should be normalized to improve stability


[N, r]=size(C);    
%C=C./repmat(sqrt(sum(C.^2, 2)), 1, r);
[r1, n, r2]=size(x0);

x=x0(:);
sigx=1./(1+exp(-(C*x.*b-1)));    
sigx=max(sigx, realmin);
tempn=reshape(permute(x0,[2 1 3]), n, r1*r2);
fx=(-1/N)*sum(log(sigx)) + gamma/2*sum(dotkron(tempn,tempn))*vn;  % The second item should not exceed REALMAX
Gn=my_Jacob(tempn);    Gn=reshape(Gn, r1, r2, n, length(vn));
Gn=permute(Gn, [1 3 2 4]);    Gn=reshape(Gn, r1*n*r2, length(vn));
g=(1/N)*C'*((sigx-1).*b) + gamma/2*Gn*vn;    
temp1=C.*repmat(sigx, 1, r);    temp2=C.*repmat(1-sigx, 1, r);
Hn=my_Jacob2(vn, n);    Hn=reshape(Hn, r1, r2, n, r1, r2, n);
Hn=permute(Hn, [1 3 2 4 6 5]);
Hn=reshape(Hn, r1*n*r2, r1*n*r2);
H=(1/N)*temp1'*temp2 + gamma/2*Hn;   

rho=0.2;    
s=-pinv(H)*g;
%s=-H\g;
fxx=Inf;   alpha=2;
while fxx > fx+rho*alpha*g'*s
    alpha=alpha/2;
    xx=x+alpha*s;
    sigxx=1./(1+exp(-(C*xx.*b-1)));    
    sigxx=max(sigxx, realmin);
    tempn=permute(reshape(xx, r1, n, r2), [2 1 3]);
    tempn=reshape(tempn, n, r1*r2);
    fxx=(-1/N)*sum(log(sigxx)) + gamma/2*sum(dotkron(tempn,tempn))*vn;  % The second item should not exceed REALMAX
end
x=xx;    fx=fxx; 
normA=sum(dotkron(tempn,tempn))*vn;


end


function  Gn=my_Jacob(tempn)
[n, L]=size(tempn);
Gn=zeros(L*n, L^2);
for i=1:n
    a=tempn(i,:);
    temp1=[];    temp2=zeros(L, L^2);
    for j=1:L
        temp1=blkdiag(temp1, a);
        temp2(:, 1+(j-1)*L:j*L)=a(j)*eye(L);
    end
    Gn(1+(i-1)*L:i*L, :)=temp1+temp2;
end

end


function  Hn=my_Jacob2(vn, n)
L=sqrt(length(vn));
temp=reshape(vn, L, L);
temp=temp+temp';
Hn=kron(eye(n), temp);


end

