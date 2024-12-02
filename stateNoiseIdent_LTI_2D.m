% State noise density identification (2D LTI)
%
% Code aligned with the paper:
%
% State Noise Density Identification for LTV System by Kernel Deconvolution}
%
% by J. Dunik, O. Kost, and O. Straka

clear; close;

% Data generation
% -- number of data
T = 1e6;
nx = 2;
nz = 2;
% -- system matrices definition
F = eye(nx);
H = eye(nx);
G = eye(nx); 
% -- control
u = [sin(1:T);cos(1:T)];
% -- noises definition
% --- state noise GS
wbar1 = [-2;2];
Q1 = [4 1.5; 1.5 2];
alpha1 = 0.5;
wbar2 = [2;-2];
Q2 = [1 0.5; 0.5 2];
%alpha2 = 1-alpha1;
% --- meansurement noise Gaussian
vbar = [0; 0];
R = [1 -0.5; -0.5 1];
% -- state and measurement generation
xi = zeros(nx,T+1);
z = zeros(nz,T);
w = zeros(nx,T);
v = zeros(nz,T);
for k=1:T
    % noise generation
    auxs = rand;
    if auxs<alpha1
        w(:,k) = wbar1 + chol(Q1)'*randn(nx,1);
    else
        w(:,k) = wbar2 + chol(Q2)'*randn(nx,1);
    end
    v(:,k) = vbar + chol(R)'*randn(nz,1);

    % state and measurement generation
    xi(:,k+1) = F*xi(:,k) + G*u(:,k) + w(:,k);
    z(:,k) = H*xi(:,k) + v(:,k);
end

% Residue calculation
r_all = zeros(nx,T-1);
nu_verification = zeros(nx,T-1);
r_verification = zeros(nx,T-1);
M = H*F*inv(H);
invH = inv(H);
for k=2:T
    zpred = M*z(:,k-1) + H*G*u(:,k-1);
    r_all(:,k-1) = invH*(z(:,k) - zpred);
    r_verification(:,k-1) = invH * (H*w(:,k-1) + v(:,k) - H*F*invH*v(:,k-1)); % for verification only
    nu_verification(:,k-1) = invH * (v(:,k) - M*v(:,k-1)); % for verification only
end

% Selection of every other data sample
rsamples = r_all(:,1:2:end);
Nsamples = length(rsamples);

% Measurement noise properties calculation (variable nu)
varNu = 2*R;%invH*R*invH' + (F*invH)*R*(F*invH)';

% Known and kernel densities calculation (of p(r) and p(v))
% -- grid points definition
N = 2*1024;
xi1 = linspace(-20, 20, N);  % Adjust the range as needed for your problem
xi2 = linspace(-20, 20, N);  % Adjust the range as needed for your problem
dxi1 = mean(diff(xi1));
dxi2 = mean(diff(xi2));
[xi1_mat, xi2_mat] = meshgrid(xi1,xi2);
xi1_vec = xi1_mat(:);
xi2_vec = xi2_mat(:);
xi = [xi1_vec, xi2_vec];
dxi = dxi1*dxi2;
% -- Construction of p(nu), which is function of p(v)
pnu_vec = mvnpdf(xi,vbar',varNu);
pnu = reshape(pnu_vec,N,N); 

% -- Construction of kernel density p(r) construction
Bbandwidth = 0.12;
[pr_kernel_vec, ~] = ksdensity(rsamples', xi, 'Bandwidth', Bbandwidth, 'BoundaryCorrection','reflection','Kernel','epanechnikov');
pr_kernel = reshape(pr_kernel_vec,N,N); 

% Characteristic functions calculation (Fourier transformation of gridded densities)
Fnu = fft2(pnu);
Fr = fft2(pr_kernel);

% CF of state noise calculation
% -- smootheness adjustment (optimised wrt state noise moments estimated by MDM)
epsilon = 1.2e4;  
% -- CF calculation using Wiener deconvolution with regularisation
Fnu_wiener = conj(Fnu) ./ (abs(Fnu).^2 + epsilon);
Fw_deconv = Fnu_wiener .* Fr;

% State noise density calculation by inverse Fourier transform
pw_deconv = fftshift(ifft2(Fw_deconv));
% pw_deconv = real(pw_deconv) / (sum(real(pw_deconv),'all') * dxi);
pw_deconv(pw_deconv<0) = 0;
% -- normalisation to ensure it sums to 1
pw_deconv = real(pw_deconv) / (sum(real(pw_deconv),'all') * dxi);

% True state noise (for comparison)
pw_true_grid = reshape(alpha1*mvnpdf(xi,wbar1',Q1) + (1-alpha1)*mvnpdf(xi,wbar2',Q2),N,N); 

% Plot the original PDFs and the deconvolved PDF
hw = histogram2(w(1,:),w(2,:),'Normalization','pdf');
xbw = hw.XBinEdges(1:end-1)+mean(diff(hw.XBinEdges(1:end-1)))/2;
ybw = hw.YBinEdges(1:end-1)+mean(diff(hw.YBinEdges(1:end-1)))/2;
pw_true = hw.Values;
hv = histogram2(v(1,:),v(2,:),'Normalization','pdf');
xbv = hv.XBinEdges(1:end-1)+mean(diff(hv.XBinEdges(1:end-1)))/2;
ybv = hv.YBinEdges(1:end-1)+mean(diff(hv.YBinEdges(1:end-1)))/2;
pv = hv.Values;
hnu = histogram2(nu_verification(1,:),nu_verification(2,:),'Normalization','pdf');
xbnu = hnu.XBinEdges(1:end-1)+mean(diff(hnu.XBinEdges(1:end-1)))/2;
ybnu = hnu.YBinEdges(1:end-1)+mean(diff(hnu.YBinEdges(1:end-1)))/2;
pnu = hnu.Values;
hr = histogram2(rsamples(1,:),rsamples(2,:),'Normalization','pdf');
xbr = hr.XBinEdges(1:end-1)+mean(diff(hr.XBinEdges(1:end-1)))/2;
ybr = hr.YBinEdges(1:end-1)+mean(diff(hr.YBinEdges(1:end-1)))/2;
pr = hr.Values;
close;
% --
figure
subplot(1,4,1)
mesh(xbw,ybw,pw_true')
zlabel('$p_w$ (true)','Interpreter','latex')
xlabel('$w_{k,1}$','Interpreter','latex')
ylabel('$w_{k,2}$','Interpreter','latex')
subplot(1,4,2)
mesh(xbv,ybv,pv')
zlabel('$p_v$','Interpreter','latex')
xlabel('$v_{k,1}$','Interpreter','latex')
ylabel('$v_{k,2}$','Interpreter','latex')
subplot(1,4,3)
mesh(xbnu,ybnu,pnu')
zlabel('$p_\nu$','Interpreter','latex')
xlabel('$\nu_{k,1}$','Interpreter','latex')
ylabel('$\nu_{k,2}$','Interpreter','latex')
subplot(1,4,4)
mesh(xbr,ybr,pr')
zlabel('$p_r$','Interpreter','latex')
xlabel('$r_{k,1}$','Interpreter','latex')
ylabel('$r_{k,2}$','Interpreter','latex')
% --
figure
subplot(1,3,1)
mesh(xi1,xi2,pw_true_grid')
zlabel('$p_w$ (true)','Interpreter','latex')
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')
subplot(1,3,2)
mesh(xi1,xi2,pw_deconv')
zlabel('$\hat{p}_w$ (estimated)','Interpreter','latex')
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')
subplot(1,3,3)
mesh(xi1,xi2,pw_true_grid'-pw_deconv')
zlabel('$\tilde{p}_w$','Interpreter','latex')
xlabel('$\xi_1$','Interpreter','latex')
ylabel('$\xi_2$','Interpreter','latex')

disp('Deconvolved pw integral error')
disp(sum(abs(pw_true_grid'-pw_deconv')*dxi,'all'))