function y=check(Input, x)

N=size(Input,1);    d=length(x);
n=zeros(1,d);     r=ones(1,d+1);
for i=1:d
    r(i)=size(x{i}, 1);    n(i)=size(x{i}, 2);
end

y=ones(N,1);    R=1;
for i=1:d
    temp=R*reshape(x{i}, r(i), n(i)*r(i+1));
    r(i)=size(R,1);
    temp=reshape(temp, r(i)*n(i), r(i+1));
    [Q, R]=qr(temp, 0);
    Mati=repmat(Input(:,i), 1, n(i)).^(kron(0:n(i)-1, ones(N,1)));
    y=dotkron(y, Mati)*Q;
end
    
y=y*R;



%y=zeros(N,1);
%for j=1:N
%    p=1;
%    for i=d:-1:1
%        temp=reshape(x{i}, r(i)*n(i), r(i+1))*p;
%        temp=reshape(temp, r(i), n(i));
%        v=Input2(j,i).^(0:n(i)-1);
%        p=temp*v';
%    end
%    y(j)=p;
%end

    
end

