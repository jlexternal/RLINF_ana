function fun_fit_model_stim_dist(samp,subj,cond,sfit,varargin)
%  FUN_FIT_MODEL_STIM_DIST  Helper function for fitting inference-free stimulus
%  categorization model to RLINF study data
%
%  Usage: FUN_FIT_MODEL_STIM_DIST(samp,subj,cond,sfit,...)
%
%  where samp is the sample name
%        subj is the subject number
%        cond is the condition number (0=bandit, 1=fairy)
%        sfit is the fitting configuration string (see below)
%
%  The function runs multiple BADS fits to localize the posterior maximum, and
%  then a single VBMC fit to estimate the posterior distribution using the best
%  BADS fit as initialization values. The function saves fitting output to files
%  in a folder (default: ./fit/stim).

% check number of input arguments
narginchk(4,inf);

% check input arguments
if ~ischar(samp)
    error('Invalid sample name!');
end
if ~ismember(cond,[0,1])
    error('Invalid condition number (0=bandit, 1=fairy)!');
end
if ~ischar(sfit) || numel(sfit) ~= 1
    error('Invalid fitting configuration string!');
end

% add toolboxes to path
addpath('~/Dropbox/MATLAB/Toolboxes/bads/');
addpath('~/Dropbox/MATLAB/Toolboxes/vbmc/');

% parse name-value input arguments
ip = inputParser;
ip.StructExpand = true; % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('pathname','./fits/stim',@(x)ischar(x)&&exist(x)==7);
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
% disable inference
cfg.h = 0.5;
cfg.gamma = 0;
% interpret fitting configuration string
switch str2num(sfit(1))
    case 0 % without stimulus distortion
        cfg.alpha = 0;
        cfg.omega = 0;
    case 1 % with stimulus distortion
        cfg.alpha = [];
        cfg.omega = [];
    otherwise
        error('Invalid fitting configuration string!');
end
cfg.sigma = [];
cfg.eta = 0; % without stimulus bias
cfg.tau = 0; % argmax policy

% set filename for fitting output
fname_out = sprintf('fit_model_stim_dist_%s_subj%03d_cond%d_%s_bads.mat', ...
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
fname_out = sprintf('fit_model_stim_dist_%s_subj%03d_cond%d_%s_vbmc.mat', ...
    samp,subj,cond,sfit);
fname_out = fullfile(pathname,fname_out);

if ~exist(fname_out,'file') || overwrite
    % fit model using VBMC
    out = fit_model_inf_dist(dat,cfg,'fitalgo','vbmc','pini',pini,varargin{:});
    % save fitting output
    save(fname_out,'out');
end

end