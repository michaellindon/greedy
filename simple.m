clc
%Create some data
n=1000;
p=100;
x=randn(n,p);
%Center and Scale Design Matrix 
for(i=1:p)
   x(:,i)=x(:,i)-mean(x(:,i));
   x(:,i)=x(:,i).*(sqrt(n)/sqrt(x(:,i)'*x(:,i)));
end
b=zeros(p,1);
b(1)=1;
b(2)=2;
b(3)=3;
b(4)=4;
b(5)=5;
phi=1;
y=x*b+sqrt(1/phi)*randn(n,1);
y=y-mean(y);


%ols for reference
Bols=linsolve(x'*x,x'*y);

%Model Parameters
priorprob=0.01*ones(p,1); %Predictor inclusion probability
logpriorprob=log(priorprob);
priorodds=priorprob./(1-priorprob); %Predictor inclusion odds
logpriorodds=log(priorodds); %Predictor inclusion log odds
lam=ones(p,1); %Penalty coming from prior on regression coefficients
Lam=diag(lam);
xx=x'*x;

gamma=zeros(p,1);
logdensity=ones(p,1)*sum(gamma.*logpriorodds);
logdensityold=logdensity;
sort_improv=ones(p,1);

fprintf('Order of egressors Added\n');
while sort_improv(1)>0 
    
for i=1:p
   gamma_prop=gamma;
   gamma_prop(i)=1;
   inc_indices=find(gamma_prop);
   Lamg=Lam(inc_indices,inc_indices);
   xxg=xx(inc_indices,inc_indices);
   xg=x(:,inc_indices);
   B=linsolve(xxg+Lamg,xg'*y);
   logdensity(i)=0.5*log(det(Lamg))-0.5*log(det(Lamg+xxg))+0.5*phi*(B'*xg'*y)+sum(gamma_prop.*logpriorodds);
end

[sort_improv,sort_indices]=sort(logdensity-logdensityold,'descend');

if(sort_improv(1)>0) 
    gamma(sort_indices(1))=1;
    disp(sort_indices(1));
end

logdensityold=ones(p,1)*logdensity(sort_indices(1));
end
fprintf('Model Selected\n')
disp(find(gamma));