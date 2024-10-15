function fun_fit_model_inf_dist(samp,subj,cond,sfit,varargin)

% check number of input arguments
narginchk(4,inf);

% check input arguments
if ~ischar(samp)
    error('Invalid sample name!');
end
if ~ismember(cond,[0,1])
    error('Invalid condition number (0=bandit, 1=fairy)!');
end
if ~ischar(sfit) || numel(sfit) ~= 5
    error('Invalid fitting configuration string!');
end

% add toolboxes to path
addpath('~/Dropbox/MATLAB/Toolboxes/bads/');
addpath('~/Dropbox/MATLAB/Toolboxes/vbmc/');

% parse name-value input arguments
ip = inputParser;
ip.StructExpand = true; % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('pathname','./fit/single',@(x)ischar(x)&&exist(x)==7);
ip.addParameter('pnul_max',0.05,@(x)isnumeric(x)&&isscalar(x)&&(x>=0&&x<=1));
ip.addParameter('overwrite',false,@(x)islogical(x)&&isscalar(x));
ip.parse(varargin{:});

% get parsed name-value input arguments
pathname  = ip.Results.pathname;  % output path name
pnul_max  = ip.Results.pnul_max;  % exclusion p-value of chance-level accuracy
overwrite = ip.Results.overwrite; % overwrite existing files?

% get data structure for model fitting
dat = dat_model_inf(samp,subj,cond);

% check whether missing data
if any(isnan(dat.resp))
    % abort fit
    fprintf('Missing data for subj%03d, cond%d!\n',subj,cond);
    return
end

% check whether poor performance
itrl = find(dat.trlnum > 1);
pnul = binocdf(nnz(dat.resp(itrl) == dat.bmstate(itrl)),numel(itrl),0.5,'upper');
if pnul > pnul_max
    % abort fit
    fprintf('Poor performance for subj%03d, cond%d!\n',subj,cond);
    return
end

% create fitting configuration structure
cfg = [];
% interpret fitting configuration string
switch str2num(sfit(1))
    case 0 % exact inference
        cfg.sigma = 0;
    case 1 % noisy inference
        cfg.sigma = [];
    otherwise
        error('Invalid fitting configuration string!');
end
switch str2num(sfit(2))
    case 0 % argmax policy
        cfg.tau = 0;
    case 1 % softmax policy
        cfg.tau = [];
    otherwise
        error('Invalid fitting configuration string!');
end
switch str2num(sfit(3))
    case 0 % without prior compression
        cfg.gamma = 0;
    case 1 % with prior compression
        cfg.gamma = [];
    otherwise
        error('Invalid fitting configuration string!');
end
switch str2num(sfit(4))
    case 0 % without stimulus distortion
        cfg.alpha = 0;
        cfg.omega = 0;
    case 1 % with stimulus distortion
        cfg.alpha = [];
        cfg.omega = [];
    otherwise
        error('Invalid fitting configuration string!');
end
switch str2num(sfit(5))
    case 0 % without stimulus bias
        cfg.eta = 0;
    case 1 % with stimulus bias
        cfg.eta = [];
    otherwise
        error('Invalid fitting configuration string!');
end

% set filename for fitting output
fname_out = sprintf('fit_model_inf_dist_%s_subj%03d_cond%d_%s_bads.mat', ...
    samp,subj,cond,sfit);
fname_out = fullfile(pathname,fname_out);

if ~exist(fname_out,'file') || overwrite
    % fit model using BADS
    out = fit_model_inf_dist(dat,cfg,'fitalgo','bads',varargin{:});
    % save fitting output
    save(fname_out,'out');
else
    % load fitting output
    load(fname_out,'out');
end

% set initial values using BADS fit
pini = [];
for i = 1:numel(out.pnam)
    pini.(out.pnam{i}) = out.(out.pnam{i});
end

% set filename for fitting output
fname_out = sprintf('fit_model_inf_dist_%s_subj%03d_cond%d_%s_vbmc.mat', ...
    samp,subj,cond,sfit);
fname_out = fullfile(pathname,fname_out);

if ~exist(fname_out,'file') || overwrite
    % fit model using VBMC
    out = fit_model_inf_dist(dat,cfg,'fitalgo','vbmc','pini',pini,varargin{:});
    % save fitting output
    save(fname_out,'out');
end

end