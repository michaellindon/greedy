%%%%%%%%%%%%%%%%%%%%%%%%Student's t Priors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Uses a CEM conditional EM algorithm to find posterior mode of            %
%(Beta,Gamma,Phi) by maximizing Phi|(Beta,Gamma) and (Beta,Gamma)|Phi.    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
%True Model (b-regression coefficients)
b=zeros(p,1);
b(1)=1;
b(2)=2;
b(3)=3;
b(4)=4;
b(5)=5;
%Inject some noise
phi=1;
y=x*b+sqrt(1/phi)*randn(n,1);
%Subtract Mean (because integrating out intercept over uniform prior)
y=y-mean(y);

%Model Parameters
alpha=1;
priorprob=0.01*ones(p,1); %Predictor inclusion probability
logpriorprob=log(priorprob);
priorodds=priorprob./(1-priorprob); %Predictor inclusion odds
logpriorodds=log(priorodds); %Predictor inclusion log odds
lam=ones(p,1); %Penalty coming from prior on regression coefficients
Lam=diag(lam);
xx=x'*x;

%Calculate (log)Density of Null Model (all up to proportionality constant)
gamma=zeros(p,1);
B=zeros(p,1);
logdensity=(-1)*ones(p,1)*exp(100);
logdensityold=logdensity;

%Begin Greedy Algorithm
sort_improv=ones(p,1);
fprintf('Order of egressors Added\n');
while sort_improv(1)>0 %While still improving the log posterior density
   
    %Construct sub-matrices
   inc_indices=find(gamma); %Included Indices
   Lamg=Lam(inc_indices,inc_indices); 
   xxg=xx(inc_indices,inc_indices);
   xg=x(:,inc_indices);
   Bg=B(inc_indices,1);
    
%%%%%%%%%%%%%%%%%%%%%%%%Begin Expectation Step%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   lam=(alpha+gamma)./(alpha+phi*gamma.*B.*B);
   Lam=diag(lam);
   Lamg=Lam(inc_indices,inc_indices); 
%%%%%%%%%%%%%%%%%%%%%%%%End Expectation Step%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


   
%%%%%%%%%%%%%%%%%%%%%%%%Begin Maximization for phi%%%%%%%%%%%%%%%%%%%%%%%%
    b=0.5*dot(y-xg*Bg,y-xg*Bg)+0.5*dot(Bg,Lamg*Bg);
	a=0.5*(n+sum(gamma)-3);
	phi=a/b;
%%%%%%%%%%%%%%%%%%%%%%%%End Maximization for phi%%%%%%%%%%%%%%%%%%%%%%%%%%%
    


%%%%%%%%%%%%%%%%%%%%%%%%Begin Maximization of (Beta,Gamma)%%%%%%%%%%%%%%%%%
%Greedy Part    
for i=1:p
   gamma_prop=gamma; %Gamma is binary vector for inclusion/exclusion
   gamma_prop(i)=1; %Proposed gamma
   inc_indices=find(gamma_prop); %Included Indices
      
   %Construct sub-matrices
   Lamg=Lam(inc_indices,inc_indices); 
   xxg=xx(inc_indices,inc_indices);
   xg=x(:,inc_indices);
   
   Bg=linsolve(xxg+Lamg,xg'*y);
   logdensity(i)=sum(gamma_prop)*log(phi)-0.5*phi*(y'*y-y'*xg*Bg)+sum(gamma_prop.*logpriorodds);
end

%Which Predictor gives greatest improvement
[sort_improv,sort_indices]=sort(logdensity-logdensityold,'descend');

%If improvement is positive, accept predictor and echo added predictor index
if(sort_improv(1)>0) 
    gamma(sort_indices(1))=1;
    Gam=diag(gamma);
    B=Gam*linsolve(Gam*xx*Gam+Lam,Gam*x'*y);
    disp(sort_indices(1));
end
%%%%%%%%%%%%%%%%%%%%%%%%End Maximization of (Beta,Gamma)%%%%%%%%%%%%%%%%%%%

logdensityold=ones(p,1)*logdensity(sort_indices(1));
end

%Report Chosen Model
fprintf('Model Selected\n')
disp(find(gamma));