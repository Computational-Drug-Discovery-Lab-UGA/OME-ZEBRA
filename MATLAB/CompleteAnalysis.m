% loading SOARS ratio estimate
clear;
load FirstHour/AnalysisResults;
clear mn* sd* sm um vm tc*;
%rt = rt(:,2160:2720);
% rt1 = rt;
% load SecondHourAnalysisResults;
% clear mn* sd* sm um vm tc*;
% rt = [rt1, rt];
% clear rt1;
N = 3590;

% calculating calcium concentration from ratio
cal = zeros(size(rt));
rmin = min(rt,[],2)-0.01;
rmax = max(rt,[],2)+0.01;
for i = 1:size(rt,2)
  cal(:,i) = 10^(-6.5) * (rt(:,i) - rmin)./(rmax - rt(:,i));
end;
clear rt;

% more accurate estimate of firing rate
dt = 1;
tau = 1.75;
alpha = 1e-5;
m = zeros([size(cal,1) size(cal,2)-1]);
for i = 1:size(m,1)
%  temp = 1/tau * (1./log(1 - (alpha * exp(-dt/tau) * (1 - exp(dt/tau)))./(cal(i,2:3590) - cal(i,1:3589) * exp(-dt/tau))));
  temp = (1 - (alpha * exp(-dt/tau) * (1 - exp(dt/tau)))./(cal(i,2:N) - cal(i,1:N-1) * exp(-dt/tau)));
  temp(find(temp < 0)) = 0;
  m(i,:) = 1/tau * (1./log(temp));
end;
clear cal;

% mask data and determine dimension
load ConnectionMask mask;
mask = reshape(mask, [128*128 1]);
idx = find(mask > 0);
[u, s, v] = svd(m(idx,:), 0);
%[u, s, v] = svd(m, 0);
u = u(:,1:200);
s = s(1:200,1:200);
v = v(:,1:200);
n = 50;
recon = u(:,1:n) * s(1:n,1:n) * v(:,1:n)';
recon = recon - min(min(recon)) + 0.01;
clear u s v;
[u1, s1, v1] = svd(recon, 0);
u2 = u1(:,1:50);
s2 = s1(1:50,1:50);
v2 = v1(:,1:50);

% non-negative factorization
rest = 0;
for i = 1:N-1
  rest = rest + (recon(:,i))' * (recon(:,i));
end;
rest = sqrt(rest)
for n2 = [20, 30, 40, 50];
  tic;
  w = rand([size(u2,1) n2]);
  h = rand([n2 size(recon,2)]);
  for i = 1:1000
    h = h .* (((w' * u2) * s2) * v2')./((w' * w) * h);
    w = w .* (u2 * (s2 * (v2' * h')))./(w * (h * h'));
  end;
  toc;
  disp('hi');
  res = 0;
  for i = 1:N-1
    res = res + (recon(:,i) - w * h(:,i))' * (recon(:,i) - w * h(:,i));
  end;
  res = sqrt(res)/rest
end;

restsvd = 0;
for i = 1:N-1
  restsvd = restsvd + recon(:,i)' * recon(:,i);
end;
restsvd = sqrt(restsvd)
for n2 = [20, 30, 40, 50]
  ressvd = 0;
  for i = 1:N-1
    ressvd = ressvd + (recon(:,i) - u2(:,1:n2) * (s2(1:n2,1:n2) * v2(i,1:n2)'))' * (recon(:,i) - u2(:,1:n2) * (s2(1:n2,1:n2) * v2(i,1:n2)'));
  end;
  ressvd = sqrt(ressvd)/restsvd; [n2 ressvd]
end;

n2 = 20;
tic;
w = rand([size(u2,1) n2]);
h = rand([n2 size(recon,2)]);
for i = 1:100
  h = h .* (((w' * u2) * s2) * v2')./((w' * w) * h);
  w = w .* (u2 * (s2 * (v2' * h')))./(w * (h * h'));
end;
toc;

w2 = zeros([size(mask,1) n2]);
w2(idx,:) = w;
clear pwr;
for i = 1:n2
  pwr(i) = w2(:,i)' * w2(:,i);
end;
[pwrSort, idxpwr] = sort(pwr, 2, 'descend');
w2 = w2(:,idxpwr);
h2 = h(idxpwr,:);

% non-negative fit
hprime = h2(:,2:N-1);
h0 = h2(:,1:N-2);
ksmallnmf = (hprime * hprime') * inv(h0 * hprime');
ksmallnmf2 = ksmallnmf - diag(diag(ksmallnmf));
for k = 1:100
  pcolor(reshape(w2(:,1), [128 128])); shading flat; colormap gray; %colorbar;
  [x, y] = ginput(1);
  j = 128 * (floor(x) - 1) + floor(y);
  subplot(1,2,1);
  temp = pinv(w2);
  pcolor(reshape(w2 * ksmallnmf2 * temp(:,j), [128 128])); 
  shading flat; colormap gray; %colorbar; %caxis([-0.05 0.05]);
  hold on;
  hand = plot(floor(x), floor(y), 'k.');
  set(hand, 'MarkerSize', 20);
  hold off;
  subplot(1,2,2);
  pcolor(reshape(w2(j,:) * ksmallnmf2 * temp, [128 128])); 
  shading flat; colormap gray;  %colorbar; %caxis([-0.05 0.05]);
  hold on;
  hand = plot(floor(x), floor(y), 'k.');
  set(hand, 'MarkerSize', 20);
  hold off;
  pause;
end;

% svd fit
vprime = s2(1:n2,1:n2) * v2(2:N-1,1:n2)';
v0 = s2(1:n2,1:n2) * v2(1:N-2,1:n2)';
ksmallsvd = (vprime * vprime') * pinv(v0 * vprime');
ksmallsvd2 = ksmallsvd - diag(diag(ksmallsvd));
temp1 = zeros([size(w2,1) n2]);
temp1(idx,:) = u2(:,1:n2);
temp1 = squeeze(temp1);
temp1 = u2(:,1:n2);
for k = 1:100
  subplot(1,2,1);
  pcolor(reshape(temp1(:,1), [128 128])); shading flat; colormap gray; %colorbar;
  [x, y] = ginput(1);
  j = 128 * (floor(x) - 1) + floor(y);
  subplot(1,2,1);
  temp = temp1 * ksmallsvd2 * temp1(j,:)';
  pcolor(reshape(temp, [128 128])); 
  shading flat; colormap gray; colorbar; %caxis([-0.05 0.05]);
  hold on;
  hand = plot(floor(x), floor(y), 'k.');
  set(hand, 'MarkerSize', 20);
  hold off;
  subplot(1,2,2);
  temp = temp1(j,:) * ksmallsvd2 * temp1';
  pcolor(reshape(temp, [128 128])); 
  shading flat; colormap gray;  colorbar; %caxis([-0.05 0.05]);
  hold on;
  hand = plot(floor(x), floor(y), 'k.');
  set(hand, 'MarkerSize', 20);
  hold off;
  pause;
end;

for j = [12107, 7740, 13274]
  x = floor(j/128);
  y = rem(j, 128);
  subplot(2,2,1);
  temp = temp1 * ksmallsvd2 * temp1(j,:)';
  pcolor(reshape(temp, [128 128])); 
  shading flat; colormap gray; colorbar; %caxis([-0.05 0.05]);
  hold on;
  axis square;
  fill([x-2 x-2 x+2 x+2], [y-2 y+2 y+2 y-2], 'k');
  hold off;
  subplot(2,2,2);
  temp = temp1(j,:) * ksmallsvd2 * temp1';
  pcolor(reshape(temp, [128 128])); 
  shading flat; colormap gray;  colorbar; %caxis([-0.05 0.05]);
  hold on;
  axis square;
  fill([x-2 x-2 x+2 x+2], [y-2 y+2 y+2 y-2], 'k');
  hold off;
  subplot(2,2,3);
  temp = pinv(w2);
  pcolor(reshape(w2 * ksmallnmf2 * temp(:,j), [128 128])); 
  shading flat; colormap gray; colorbar; %caxis([-0.05 0.05]);
  hold on;
  axis square;
  fill([x-2 x-2 x+2 x+2], [y-2 y+2 y+2 y-2], 'k');
  hold off;
  subplot(2,2,4);
  pcolor(reshape(w2(j,:) * ksmallnmf2 * temp, [128 128])); 
  shading flat; colormap gray;  colorbar; %caxis([-0.05 0.05]);
  hold on;
  axis square;
  fill([x-2 x-2 x+2 x+2], [y-2 y+2 y+2 y-2], 'k');
  hold off;
  eval(['print -dtiff -r300 PrePostSynaptic', int2str(j), '.tif;']);
  pause;
end;

u2 = zeros(size(mask,1), n2);
u2(idx,:) = u1(:,1:n2);
[vec, val] = eig(ksmallsvd);
for i = 1:50
  pcolor(real(reshape(u2 * vec(:,i), [128 128]))); shading flat; colormap gray;
  axis square; axis off;
  drawnow;
  eval(['print -dtiff -r300 EigvecReal', int2str(i), '.tif;']);
  pcolor(imag(reshape(u2 * vec(:,i), [128 128]))); shading flat; colormap gray;
  axis square; axis off;
  drawnow;
  eval(['print -dtiff -r300 EigvecImag', int2str(i), '.tif;']);
end;

hst = zeros([1 3000]);
for i = 1:size(m,1)
  for j = 1:size(m,2)
    hst(ceil(m(i,j)*1000)+1) = hst(ceil(m(i,j)*1000)+1) + 1;
  end;
end;
