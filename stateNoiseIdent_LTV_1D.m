% State noise density identification (1D LTV)
%
% Code aligned with the paper:
%
% State Noise Density Identification for LTV System by Kernel Deconvolution}
%
% by J. Dunik, O. Kost, and O. Straka

clear; close;

% Data generation
% -- number of data
T = 4e6+1;
% -- system matrices definition
%F = 0.9*ones(1,T); % LTI
F = 0.9*sin([1:T]*0.0001); % LTV
H = 1*ones(1,T);
G = cos(1:T); % LTV
% -- control
u = ones(1,T);
% -- noises definition and generation
w = random('Rayleigh',2,1,T);
% w = raylrnd(0.5,1,T);
R = 1;
v = sqrt(R).*randn(1,T);
% -- state and measurement generation
xi = zeros(1,T+1);
z = zeros(1,T);
for k=1:T
    xi(k+1) = F(k)*xi(k) + G(k)*u(k) + w(k);
    z(k) = H(k)*xi(k) + v(k);
end

% Residue calculation
r_all = zeros(1,T-1);
%r_verification = zeros(1,T-1);
nu_verification = zeros(1,T-1);
pinvH = zeros(1,T);
pinvH(1) = pinv(H(1));
for k=2:T
    pinvH(k) = pinv(H(k));
    zpred = H(k)*F(k-1)*pinv(H(k-1))*z(k-1) + H(k)*G(k-1)*u(k-1);
    r_all(k-1) = pinvH(k)*(z(k) - zpred);
    %r_verification(k-1) = pinvH(k) * (H(k)*w(k-1) + v(k) - H(k)*F(k-1)*pinvH(k-1)*v(k-1)); % for verification only
    nu_verification(k-1) = pinvH(k) * (v(k) - H(k)*F(k-1)*pinvH(k-1)*v(k-1)); % for verification only
end

% Selection of every other data sample
rsamples = r_all(1:2:end);
Nsamples = length(rsamples);

% Measurement noise properties calculation (variable nu)
varNu_all = zeros(1,T-1);
for k=2:T
    varNu_all(k-1) = pinvH(k)*R*pinvH(k)' + (F(k-1)*pinvH(k-1))*R*(F(k-1)*pinvH(k-1))';
end
varNu = varNu_all(1:2:end);

% Known and kernel densities calculation (of p(r) and p(v))
% -- grid points definition
N = 2*1024;   % number of points for densities evaluation (should be a power of 2 for FFT efficiency)
xi = linspace(-10, 10, N);  
dxi = mean(diff(xi));
% -- Construction of p(nu), which is function of p(v)
pnu = zeros(1,N);
for i=1:N % evaluate p(xi) sum in all points
    pnu(i) = mean(normpdf(xi(i),zeros(1,Nsamples),sqrt(varNu)));
end
% -- Construction of kernel density p(r) construction
Bbandwidth = 0.01;
[pr_kernel, ~] = ksdensity(rsamples, xi, 'Bandwidth', Bbandwidth);

% Characteristic functions calculation (Fourier transformation of gridded densities)
Fnu = fft(pnu);
Fr = fft(pr_kernel);

% CF of state noise calculation
% -- smootheness adjustment (optimised wrt state noise moments estimated by MDM)
epsilon = 10;  
% -- CF calculation using Wiener deconvolution with regularisation
Fnu_wiener = conj(Fnu) ./ (abs(Fnu).^2 + epsilon);
Fw_deconv = Fnu_wiener .* Fr;

% State noise density calculation by inverse Fourier transform
pw_deconv = fftshift(ifft(Fw_deconv));
% -- normalisation to ensure it sums to 1
pw_deconv = real(pw_deconv) / (sum(real(pw_deconv)) * dxi);  % Ensure it sums to 1 to be a valid PDF
% -- selection of well-identified part of the density (omitting oscilations, ensuring positive values)
pw_deconv_sel = abs(pw_deconv(N/2-50:end));
pw_deconv_sel = pw_deconv_sel / (sum(pw_deconv_sel) * dxi);
xi_sel = xi(N/2-50:end);

% Plot the original PDFs and the deconvolved PDF
hw = histogram(w,'Normalization','pdf');
xbw = hw.BinEdges(1:end-1)+mean(diff(hw.BinEdges(1:end-1)))/2;
pw_true = hw.Values;
hr = histogram(rsamples,'Normalization','pdf');
xbr = hr.BinEdges(1:end-1)+mean(diff(hr.BinEdges(1:end-1)))/2;
pr_samples = hr.Values;
hv = histogram(v,'Normalization','pdf');
xbv = hv.BinEdges(1:end-1)+mean(diff(hv.BinEdges(1:end-1)))/2;
pv = hv.Values;
hnu = histogram(nu_verification,'Normalization','pdf');
xbnu = hnu.BinEdges(1:end-1)+mean(diff(hnu.BinEdges(1:end-1)))/2;
pnu = hnu.Values;
close;
% --
figure
subplot(1,3,1)
plot(xbv, pv, ':', 'DisplayName', '$p_v$','LineWidth',2);
hold on
grid on
plot(xbnu, pnu, ':', 'DisplayName', '$p_\nu$ (function of $p_v$)','LineWidth',2);
plot(xbr,pr_samples, 'DisplayName', '$p_{r}$','LineWidth',2);
plot(xbw,pw_true,'-.', 'DisplayName', '$p_w$ (true)','LineWidth',2);
%plot(xi_sel, pw_deconv_sel,'--', 'DisplayName', '$\hat{p}_w$ (estimated)','LineWidth',2);
legend;
xlabel('$\xi$')
ylabel('PDF Values')
subplot(1,3,2)
plot(xi_sel,pdf('Rayleigh',xi_sel,2), '-.', 'DisplayName', '${p}_w (true)$','LineWidth',2);
hold on
plot(xi_sel,pw_deconv_sel, 'DisplayName', '$\hat{p}_w$ (estimated)','LineWidth',2);
grid on
legend;
xlabel('$\xi$')
ylabel('PDF Values')
subplot(1,3,3)
plot(xi_sel,pdf('Rayleigh',xi_sel,2)-pw_deconv_sel, 'DisplayName', '$\tilde{p}_w$','LineWidth',2);
grid on
xlabel('$\xi$')
ylabel('$\tilde{p}_w$')

disp('Deconvolved pw integral error')
disp(sum(abs(pdf('Rayleigh',xi_sel,2)-pw_deconv_sel)*dxi))