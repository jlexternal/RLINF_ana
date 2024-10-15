function [out] = fit_model_inf_dist(dat,varargin)

% check number of input arguments
narginchk(1,inf);

% check data structure
assert(isfield(dat,'trlnum') && ...
    isnumeric(dat.trlnum) && isvector(dat.trlnum));
assert(isfield(dat,'stim') && ...
    isnumeric(dat.stim) && isvector(dat.stim));
assert(isfield(dat,'resp') && ...
    isnumeric(dat.resp) && isvector(dat.resp));
assert(isfield(dat,'efac') && ...
    isnumeric(dat.efac) && isscalar(dat.efac));

% parse name-value arguments
ip = inputParser;
ip.StructExpand = true; % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('fittype','sumlog',@(x)ischar(x)&&ismember(x,{'sumlog','logsum'}));
ip.addParameter('fitalgo','vbmc',@(x)ischar(x)&&ismember(x,{'bads','vbmc'}));
ip.addParameter('nsmp',1e3,@(x)isnumeric(x)&&isscalar(x));
ip.addParameter('nres',1e2,@(x)isnumeric(x)&&isscalar(x));
ip.addParameter('nrun',1e1,@(x)isnumeric(x)&&isscalar(x));
ip.addParameter('useprior',true,@(x)islogical(x)&&isscalar(x));
ip.addParameter('verbose',0,@(x)isscalar(x)&&ismember(x,[0,1,2]));
ip.parse(varargin{:});

% create configuration structure
cfg = merge_struct(ip.Results,ip.Unmatched);

% read data structure
cond   = dat.cond;      % condition (0=bandit, 1=fairy)
trlnum = dat.trlnum(:); % trial number
stim   = dat.stim(:);   % stimulus direction
resp   = dat.resp(:);   % response direction
rprv   = dat.rprv(:);   % previous response direction
efac   = dat.efac;      % evidence factor

% get number of trials
ntrl = numel(trlnum);

% set infinitesimal probability
peps = 0.1/ntrl;

% read configuration structure
fittype  = cfg.fittype;  % fit type (sumlog or logsum)
fitalgo  = cfg.fitalgo;  % fitting algorithm (bads or vbmc)
nsmp     = cfg.nsmp;     % number of samples used by particle filter (bads/vbmc)
nres     = cfg.nres;     % number of resamples used by fitting algorithm (bads/vbmc)
nrun     = cfg.nrun;     % number of random starting points (bads)
useprior = cfg.useprior; % use prior distributions on parameter values? (vbmc)
verbose  = cfg.verbose;  % fitting output display level (0=none 1=final 2=iter)

% convert fit type to integer for speed (1=sumlog, 2=logsum)
fittype = find(cellfun(@(s)strcmp(s,fittype),{'sumlog','logsum'}));

% define model parameters
pnam = cell(1,7); % name
pmin = nan(1,7);  % minimum value
pmax = nan(1,7);  % maximum value
pini = nan(1,7);  % initial value
pplb = nan(1,7);  % plausible lower bound
ppub = nan(1,7);  % plausible upper bound
pfun = cell(1,7); % prior function
% 1/ hazard rate ~ unif(0.001,0.999)
pnam{1} = 'h';
pmin(1) = 0.001;
pmax(1) = 0.999;
pini(1) = 0.5;
pplb(1) = 0.1;
ppub(1) = 0.9;
pfun{1} = @(x)unifpdf(x,0.001,0.999);
% 2/ prior compression ~ normal(0,1)
pnam{2} = 'gamma';
pmin(2) = -10;
pmax(2) = +10;
pini(2) = 0;
pplb(2) = -2;
ppub(2) = +2;
pfun{2} = @(x)normpdf(x,0,1);
% 3/ stimulus compression ~ exp(1)
pnam{3} = 'alpha';
pmin(3) = 0;
pmax(3) = 10;
pini(3) = 1;
pplb(3) = 0.2;
ppub(3) = 5;
pfun{3} = @(x)exppdf(x,1);
% 4/ stimulus offset ~ normal(0,0.1)
pnam{4} = 'omega';
pmin(4) = -1;
pmax(4) = +1;
pini(4) = 0;
pplb(4) = -0.2;
ppub(4) = +0.2;
pfun{4} = @(x)normpdf(x,0,0.1);
% 5/ inference noise ~ exp(1)
pnam{5} = 'sigma';
pmin(5) = 0;
pmax(5) = 10;
pini(5) = 1;
pplb(5) = 0.2;
ppub(5) = 5;
pfun{5} = @(x)exppdf(x,1);
% 6/ stimulus bias ~ normal(0,0.5)
pnam{6} = 'eta';
pmin(6) = -5;
pmax(6) = +5;
pini(6) = 0;
pplb(6) = -1;
ppub(6) = +1;
pfun{6} = @(x)normpdf(x,0,0.5);
% 7/ policy temperature ~ exp(1)
pnam{7} = 'tau';
pmin(7) = 0;
pmax(7) = 10;
pini(7) = 1;
pplb(7) = 0.2;
ppub(7) = 5;
pfun{7} = @(x)exppdf(x,1);

% set number of parameters
npar = numel(pnam);

if ~useprior
    % use flat prior for all parameters
    pfun = cell(1,npar);
end

% apply user-defined initialization values
if isfield(cfg,'pini')
    for i = 1:npar
        if isfield(cfg.pini,pnam{i}) && ~isnan(cfg.pini.(pnam{i}))
            pini(i) = cfg.pini.(pnam{i});
            % clamp initialization value within plausible bounds
            pini(i) = min(max(pini(i),pplb(i)+1e-6),ppub(i)-1e-6);
        end
    end
end

% define fixed parameters
pfix = cell(1,npar);
for i = 1:npar
    if isfield(cfg,pnam{i}) && ~isempty(cfg.(pnam{i}))
        pfix{i} = min(max(cfg.(pnam{i}),pmin(i)),pmax(i));
    end
end

% define free parameters
ifit = cell(1,npar);
pfit_ini = [];
pfit_min = [];
pfit_max = [];
pfit_plb = [];
pfit_pub = [];
n = 1;
for i = 1:npar
    if isempty(pfix{i}) % free parameter
        ifit{i} = n;
        pfit_ini = cat(2,pfit_ini,pini(i));
        pfit_min = cat(2,pfit_min,pmin(i));
        pfit_max = cat(2,pfit_max,pmax(i));
        pfit_plb = cat(2,pfit_plb,pplb(i));
        pfit_pub = cat(2,pfit_pub,ppub(i));
        n = n+1;
    end
end

% set number of fitted parameters
nfit = length(pfit_ini);

% configure prior function approximation
upfun_opt = @(x,h)x+log((1-h)./h+exp(-x))-log((1-h)./h+exp(+x)); % optimal function
upfun_hat = @(x,a,b)erf(a*x)*b; % approximate function
if exist('./fit_upfun.mat','file')
    % load best-fitting parameters from file
    load('./fit_upfun.mat','hvec','avec','bvec');
else
    % get best-fitting parameters
    hvec = (0.001:0.002:0.999)'; % hazard rates
    avec = nan(size(hvec)); % best-fitting slopes
    bvec = nan(size(hvec)); % best-fitting asymptotes
    xval = -10:0.01:+10;
    for i = 1:numel(hvec)
        % fit approximate function parameters wrt MSE
        phat = fminsearch(@(p)sum((upfun_opt(xval,hvec(i))-upfun_hat(xval,p(1),p(2))).^2),[1,1]);
        avec(i) = phat(1); % best-fitting slope
        bvec(i) = phat(2); % best-fitting asymptote
    end
    % save best-fitting parameters to file
    save('./fit_upfun.mat','hvec','avec','bvec');
end

% define stimulus function
stimfun = @(x,alpha,omega)tanh(exp(alpha)*(atanh(x)-omega));

% is the objective function noisy?
isnoisy = isempty(pfix{5}) || pfix{5} > 0;

if nfit > 0
    
    switch fitalgo
        
        case 'bads'
            % fit model using Bayesian Adaptive Direct Search
            if ~exist('bads','file')
                error('BADS missing from path!');
            end

            % configure BADS
            options = bads('defaults');
            options.UncertaintyHandling = isnoisy; % noisy objective function?
            options.NoiseFinalSamples = nres; % number of samples
            switch verbose % display level
                case 0, options.Display = 'none';
                case 1, options.Display = 'final';
                case 2, options.Display = 'iter';
            end
            
            % fit model using multiple random starting points
            fval   = nan(1,nrun);
            xhat   = cell(1,nrun);
            output = cell(1,nrun);
            for irun = 1:nrun
                done = false;
                while ~done
                    % set random starting point
                    n = 1;
                    for i = 1:npar
                        if isempty(pfix{i}) % free parameter
                            % sample starting point uniformly between plausible bounds
                            pfit_ini(n) = unifrnd(pplb(i),ppub(i));
                            n = n+1;
                        end
                    end
                    % fit model using BADS
                    [xhat{irun},fval(irun),exitflag,output{irun}] = ...
                        bads(@(x)getnl(x), ...
                        pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,[],options);
                    if exitflag > 0
                        done = true;
                    end
                end
            end
            ll_run    = -fval;
            ll_sd_run = cellfun(@(s)getfield(s,'fsd'),output);
            xhat_run  = xhat;
            % find best fit among random starting points
            [fval,irun] = min(fval);
            xhat   = xhat{irun};
            output = output{irun};
            
            % get best-fitting values
            phat = getpval(xhat);
            
            % create output structure with best-fitting values
            out = cell2struct(phat(:),pnam(:));

            % store full list of model parameters
            out.pnam = pnam;
            
            % store fitting information
            out.fittype = fittype; % fit type
            out.fitalgo = fitalgo; % fitting algorithm
            out.nsmp    = nsmp;    % number of samples used by particle filter
            out.nres    = nres;    % number of validation resamples
            out.nrun    = nrun;    % number of random starting points
            out.ntrl    = ntrl;    % number of trials
            out.nfit    = nfit;    % number of fitted parameters
            
            % get maximum log-likelihood
            out.ll = -output.fval; % estimated log-likelihood
            out.ll_sd = output.fsd; % estimated s.d. of log-likelihood
            
            % get complexity-penalized fitting metrics
            out.aic = -2*out.ll+2*nfit+2*nfit*(nfit+1)/(ntrl-nfit+1); % AIC
            out.bic = -2*out.ll+nfit*log(ntrl); % BIC
            
            % get parameter values
            out.xnam = pnam(cellfun(@isempty,pfix));
            out.xhat = xhat;
            
            % get run-specific output
            out.ll_run    = ll_run(:); % estimated log-likelihood
            out.ll_sd_run = ll_sd_run(:); % estimated s.d. of log-likelihood
            out.xhat_run  = cat(1,xhat_run{:}); % parameter values
            
            % store additional output from BADS
            out.options = options;
            out.output = output;
            
        case 'vbmc'
            % fit model using Variational Bayesian Monte Carlo
            if ~exist('vbmc','file')
                error('VBMC missing from path!');
            end
            
            % configure VBMC
            options = vbmc('defaults');
            options.MaxIter = 300; % maximum number of iterations
            options.MaxFunEvals = 500; % maximum number of function evaluations
            options.SpecifyTargetNoise = isnoisy; % noisy log-posterior function?
            switch verbose % display level
                case 0, options.Display = 'none';
                case 1, options.Display = 'final';
                case 2, options.Display = 'iter';
            end
            
            % fit model using VBMC
            [vp,elbo,~,exitflag,output] = vbmc(@(x)getlp(x), ...
                pfit_ini,pfit_min,pfit_max,pfit_plb,pfit_pub,options);
            
            % generate 10^6 samples from the variational posterior
            xsmp = vbmc_rnd(vp,1e6);
            
            % get sample statistics
            xmap = vbmc_mode(vp); % posterior mode
            xavg = mean(xsmp,1); % posterior mean
            xstd = std(xsmp,[],1); % posterior s.d.
            xcov = cov(xsmp); % posterior covariance matrix
            xmed = median(xsmp,1); % posterior medians
            xqrt = quantile(xsmp,[0.25,0.75],1); % posterior 1st and 3rd quartiles
            
            % get full parameter set with best-fitting values
            phat_map = getpval(xmap); % posterior mode
            phat_avg = getpval(xavg); % posterior mean
            
            % use posterior average as default
            phat = phat_avg;
            
            % create output structure
            out = cell2struct(phat(:),pnam(:));

            % store full list of model parameters
            out.pnam = pnam;
            
            % create substructure with posterior mode
            out.pmap = cell2struct(phat_map(:),pnam(:));
            
            % create substructure with posterior mean
            out.pavg = cell2struct(phat_avg(:),pnam(:));
            
            % store fitting information
            out.fittype = fittype; % fit type
            out.fitalgo = fitalgo; % fitting algorithm
            out.nsmp    = nsmp;    % number of samples used by particle filter
            out.nres    = nres;    % number of bootstrap resamples
            out.ntrl    = ntrl;    % number of trials
            out.nfit    = nfit;    % number of fitted parameters
            
            % store variational posterior solution
            out.vp = vp;
            
            % get ELBO (expected lower bound on log-marginal likelihood)
            out.elbo = elbo; % estimate
            out.elbo_sd = output.elbo_sd; % standard deviation
            
            % get maximum log-posterior and maximum log-likelihood
            out.lp = getlp(xmap); % log-posterior
            out.ll = getll(phat_map{:}); % log-likelihood
            
            % get parameter values
            out.xnam = pnam(cellfun(@isempty,pfix)); % fitted parameters
            out.xmap = xmap; % posterior mode
            out.xavg = xavg; % posterior mean
            out.xstd = xstd; % posterior s.d.
            out.xcov = xcov; % posterior covariance matrix
            out.xmed = xmed; % posterior median
            out.xqrt = xqrt; % posterior 1st and 3rd quartiles
            
            % store extra VBMC output
            out.options = options;
            out.output = output;
            
        otherwise
            error('Undefined fitting algorithm!');
            
    end
    
else
    
    % use fixed parameter values
    phat = getpval([]);
    
    % create output structure
    out = cell2struct(phat(:),pnam(:));
    
    % store simulation information
    out.nsmp = nsmp; % number of simulations
    out.ntrl = ntrl; % number of trials
    
    % run particle filter
    [pt_hat,xt_hat,xu_hat] = getp(phat{:});
    
    % average trajectories
    pt_avg = mean(pt_hat,2);
    xt_avg = mean(xt_hat,2);
    xu_avg = mean(xu_hat,2);

    % store trajectories
    out.pt = pt_avg; % response probabilities
    out.xt = xt_avg; % posterior belief
    out.xu = xu_avg; % *unfiltered* posterior belief

    % simulate responses
    [out.rs,out.xs] = simr(phat{:});
    
end

% store configuration structure
out.cfg = cfg;

% store prior function
out.upfun_opt = upfun_opt; % optimal function
out.upfun_hat = upfun_hat; % approximate function
out.hvec      = hvec; % hazard rates (optimal function)
out.avec      = avec; % slopes (approximate function)
out.bvec      = bvec; % asymptotes (approximate function)

% store stimulus function
out.stimfun  = stimfun; % stimulus function
out.evid_opt = stim*efac; % optimal evidence
out.evid_hat = stimfun(stim,out.alpha,out.omega)*efac; % distorted evidence

    % Get parameter set
    function [pval] = getpval(p)
        % get parameter values
        pval = cell(1,npar);
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                pval{k} = p(ifit{k});
            else % fixed parameter
                pval{k} = pfix{k};
            end
        end
    end

    % Get negative log-likelihood estimate
    function [nl] = getnl(p)
        % get parameter values
        pval = getpval(p);
        % get negative log-likelihood
        nl = -getll(pval{:});
    end

    % Get log-posterior estimate
    function [lp,lp_sd] = getlp(p)
        % get parameter values
        pval = getpval(p);
        % get log-prior
        l0 = 0;
        for k = 1:npar
            if isempty(pfix{k}) % free parameter
                if isempty(pfun{k}) % use flat prior
                    l0 = l0+log(unifpdf(pval{k},pmin(k),pmax(k)));
                else % use specified prior
                    l0 = l0+log(pfun{k}(pval{k}));
                end
            end
        end
        % get log-likelihood
        [ll,ll_sd] = getll(pval{:});
        % get log-posterior
        lp = ll+l0; % estimate
        lp_sd = ll_sd; % bootstrap s.d.
    end

    % Get log-likelihood estimate
    function [ll,ll_sd] = getll(varargin)
        % compute response probability
        p = getp(varargin{:});
        if nargout > 1
            % compute log-likelihood s.d.
            lres = nan(nres,1);
            for ires = 1:nres
                jres = randsample(nsmp,nsmp,true);
                pres = mean(p(:,jres),2);
                if fittype == 1
                    % ll = sum(log(p))
                    lres(ires) = ...
                        sum(log(pres(resp == 1)))+ ...
                        sum(log(1-pres(resp == 2)));
                elseif fittype == 2
                    % ll = log(sum(p))
                    lres(ires) = log(( ...
                        sum(pres(resp == 1))+ ...
                        sum(1-pres(resp == 2)))/ntrl)*ntrl;
                end
            end
            ll_sd = max(std(lres),1e-6);
        end
        % compute log-likelihood
        p = mean(p,2);
        p = peps+(1-peps*2)*p;
        if fittype == 1
            % ll = sum(log(p))
            ll = ...
                sum(log(p(resp == 1)))+ ...
                sum(log(1-p(resp == 2)));
        elseif fittype == 2
            % ll = log(sum(p))
            ll = log(( ...
                sum(p(resp == 1))+ ...
                sum(1-p(resp == 2)))/ntrl)*ntrl;
        end
    end

    % Get response probabilities
    function [pt,xt,xu] = getp(h,gamma,alpha,omega,sigma,eta,tau)
        % compute prior function parameters
        a_hat = interp1(hvec,avec,h,'pchip')*exp(gamma);
        b_hat = interp1(hvec,bvec,h,'pchip');
        % run particle filter
        xt = nan(ntrl,nsmp); % posterior belief
        xu = nan(ntrl,nsmp); % *unfiltered* posterior belief
        pt = nan(ntrl,nsmp); % response probabilities
        for itrl = 1:ntrl
            % perform inference
            if cond == 0 % bandit task
                if rprv(itrl) == 1
                    evid = +stimfun(+stim(itrl),alpha,omega)*efac;
                else
                    evid = -stimfun(-stim(itrl),alpha,omega)*efac;
                end
            else % fairy task
                evid = stimfun(stim(itrl),alpha,omega)*efac;
            end
            if trlnum(itrl) == 1
                xt(itrl,:) = evid;
            else
                xt(itrl,:) = evid+upfun_hat(xt(itrl-1,:),a_hat,b_hat);
            end
            xu(itrl,:) = xt(itrl,:);
            xt(itrl,:) = normrnd(xt(itrl,:),sigma);
            % apply policy
            pt(itrl,:) = 1./(1+exp(-(xt(itrl,:)+evid*eta)/tau));
            pt(itrl,isnan(pt(itrl,:))) = 0.5;
            if fittype == 1
                % run bootstrap resampling
                if resp(itrl) == 1
                    wt = pt(itrl,:);
                else
                    wt = 1-pt(itrl,:);
                end
                if nnz(wt) == 0
                    wt = ones(1,nsmp)/nsmp;
                else
                    wt = wt/sum(wt);
                end
                ismp = randsample(nsmp,nsmp,true,wt);
                xt(itrl,:) = xt(itrl,ismp);
            end
        end
    end

    % Simulate responses
    function [rs,xs] = simr(h,gamma,alpha,omega,sigma,eta,tau)
        % compute prior function parameters
        a_hat = interp1(hvec,avec,h,'pchip')*exp(gamma);
        b_hat = interp1(hvec,bvec,h,'pchip');
        % run simulations
        xs = nan(ntrl,nsmp); % posterior belief
        ps = nan(ntrl,nsmp); % response probabilities
        rs = nan(ntrl,nsmp); % simulated responses
        for itrl = 1:ntrl
            % perform inference
            if cond == 0 % bandit task
                if trlnum(itrl) == 1
                    i1 = rand(1,nsmp) < 0.5;
                else
                    i1 = rs(itrl-1,:) == 1;
                end
                i2 = ~i1;
                evid = nan(1,nsmp);
                evid(i1) = +stimfun(+stim(itrl),alpha,omega)*efac;
                evid(i2) = -stimfun(-stim(itrl),alpha,omega)*efac;
            else % fairy task
                evid = stimfun(stim(itrl),alpha,omega)*efac;
            end
            if trlnum(itrl) == 1
                xs(itrl,:) = evid;
            else
                xs(itrl,:) = evid+upfun_hat(xs(itrl-1,:),a_hat,b_hat);
            end
            xs(itrl,:) = normrnd(xs(itrl,:),sigma);
            % apply policy
            ps(itrl,:) = 1./(1+exp(-(xs(itrl,:)+evid*eta)/tau));
            rs(itrl,:) = 1+(rand(1,nsmp) > ps(itrl,:));
        end
    end

end