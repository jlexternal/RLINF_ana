function [is_miss,is_stim,is_hinf,is_null] = get_fit_stim_hinf(samp,subjlist,varargin)

% check number of input/output arguments
narginchk(2,inf);
nargoutchk(1,4);

% parse name-value input arguments
ip = inputParser;
ip.StructExpand = true; % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('pathname','./fits/merged',@(x)ischar(x)&&exist(x)==7);
ip.addParameter('pnul_typ','acc',@(x)ischar(x)&&ismember(x,{'acc','fit'}));
ip.addParameter('pnul_thr',0.05,@(x)isnumeric(x)&&isscalar(x)&&(x>=0&&x<=1));
ip.parse(varargin{:});

% get parsed name-value input arguments
pathname = ip.Results.pathname; % path name for merged fits
pnul_typ = ip.Results.pnul_typ; % defining metric for random behavior (acc or fit)
pnul_thr = ip.Results.pnul_thr; % threshold p-value for random behavior

% load merged fits of stimulus categorization model
fname = sprintf('fit_model_stim_dist_%s_1.mat',samp);
fname = fullfile(pathname,fname);
if ~exist(fname,'file')
    error('%s: file not found!',fname);
end
load(fname,'out_vbmc');
try
    elbo = cellfun(@(s)getfield(s,'elbo'),out_vbmc(subjlist,:));
catch
    nsubj = numel(subjlist);
    elbo = nan(nsubj,2);
    for isubj = 1:nsubj
        for icond = 1:2
            if ~isempty(out_vbmc{subjlist(isubj),icond})
                elbo(isubj,icond) = out_vbmc{subjlist(isubj),icond}.elbo;
            end
        end
    end
end

% load merged fits of hidden-state inference model
fname = sprintf('fit_model_inf_dist_%s_10111.mat',samp);
fname = fullfile(pathname,fname);
if ~exist(fname,'file')
    error('%s: file not found!',fname);
end
load(fname,'out_vbmc','condlist');
try
    elbo = cat(3,elbo,cellfun(@(s)getfield(s,'elbo'),out_vbmc(subjlist,:)));
    h = cellfun(@(s)getfield(s,'h'),out_vbmc(subjlist,:));
catch
    nsubj = numel(subjlist);
    elbo_tmp = nan(nsubj,2);
    h = nan(nsubj,2);
    for isubj = 1:nsubj
        for icond = 1:2
            if ~isempty(out_vbmc{subjlist(isubj),icond})
                elbo_tmp(isubj,icond) = out_vbmc{subjlist(isubj),icond}.elbo;
                h(isubj,icond) = out_vbmc{subjlist(isubj),icond}.h;
            end
        end
    end
    elbo = cat(3,elbo,elbo_tmp);
end

% label subjects
is_miss = any(isnan(elbo),3); % missing data
is_stim = ~is_miss & elbo(:,:,1) > elbo(:,:,2) | h > 0.5; % stimulus categorization
is_hinf = ~is_miss & ~is_stim; % hidden-state inference

if nargout > 3

    % compute probability of random behavior
    switch pnul_typ
        case 'acc' % wrt objective accuracy
            p_nul = nan(numel(subjlist),2);
            for isubj = 1:numel(subjlist)
                for icond = 1:2
                    dat = dat_model_inf(samp,subjlist(isubj),condlist(icond));
                    itrl = find(dat.trlnum > 1);
                    ntrl = numel(itrl);
                    p_nul(isubj,icond) = ...
                        binocdf(nnz(dat.resp(itrl) == dat.bmstate(itrl)),ntrl,0.5,'upper');
                end
            end
        case 'fit' % wrt best-fitting model
            load(fname,'out_bads');
            l_fit = -0.5*cellfun(@(s)getfield(s,'aic'),out_bads(subjlist,:));
            l_nul = cellfun(@(s)getfield(s,'ntrl'),out_bads(subjlist,:))*log(0.5);
            p_nul = 1./(1+exp(-(l_nul-l_fit)));
    end

    % relabel subjects accounting for random behavior
    is_null = ~is_miss & p_nul > pnul_thr;
    is_stim(is_null) = false;
    is_hinf(is_null) = false;

end

end