function [bs,br] = fit_bs(dat,varargin)

% parse name-value arguments
ip = inputParser;
ip.StructExpand = true; % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('nrun',1,@(x)isnumeric(x)&&isscalar(x)&&mod(x,1)==0);
ip.addParameter('nbin',10,@(x)isnumeric(x)&&isscalar(x)&&mod(x,1)==0);
ip.addParameter('sigma',inf,@(x)isnumeric(x)&&isscalar(x)&&x>0);
ip.addParameter('peps',1e-3,@(x)isnumeric(x)&&isscalar(x)&&peps>0&&peps<1);
ip.parse(varargin{:});

% get name-value arguments
nrun  = ip.Results.nrun;  % number of fitting runs
nbin  = ip.Results.nbin;  % number of bins
sigma = ip.Results.sigma; % prior standard deviation
peps  = ip.Results.peps;  % infinitesimal response probability

% get data
cond   = dat.cond;   % condition number
trlnum = dat.trlnum; % trial number
stim   = dat.stim;   % stimulus direction (in [-1,+1] range)
resp   = dat.resp;   % response direction (binary)
rprv   = dat.rprv;   % previous response direction (only used for bandit task)

% get number of trials
ntrl = numel(stim);

% set stimulus bins
ebin = linspace(-1,+1,nbin+1);
ebin = [-inf,ebin(2:nbin),+inf];
sbin = discretize(stim,ebin);

% set parameter ranges
pfit_ini = zeros(1+nbin,1);
pfit_min = zeros(1+nbin,1)-10;
pfit_max = zeros(1+nbin,1)+10;

xhat = nan(nrun,1+nbin); % best-fitting parameters
fval = nan(nrun,1); % objective function value
for irun = 1:nrun
    if nrun > 1
        % randomize starting point
        pfit_ini = normrnd(zeros(1+nbin,1),sigma);
    end

    % set fitting options
    options = optimset('fmincon');
    options = optimset(options,'Algorithm','interior-point', ...
        'FunValCheck','on','TolX',1e-12,'MaxFunEvals',1e6,'Display','notify');

    % fit logistic regression model
    [xhat(irun,:),fval(irun)] = fmincon(@(p)-getlp(p),pfit_ini,[],[],[],[],pfit_min,pfit_max,[],options);

end
if nrun > 1
    % select best fit
    [~,irun] = min(fval);
    xhat = xhat(irun,:);
end

% get best-fitting parameters
bs = xhat(1:nbin); % stimulus weights
br = xhat(1+nbin); % response weight

    % Compute model log-posterior
    function [lp] = getlp(x)
        bs_ = x(1:nbin); % stimulus weights
        br_ = x(1+nbin); % response weight
        q = nan(ntrl,1); % response Q-value
        p = nan(ntrl,1); % response probability
        for itrl = 1:ntrl
            % add stimulus weight to Q-value
            if cond == 0
                % bandit task
                if rprv(itrl) == 1
                    qs = +bs_(sbin(itrl));
                else
                    qs = -bs_(nbin-sbin(itrl)+1);
                end
            else
                % fairy task
                qs = bs_(sbin(itrl));
            end
            q(itrl) = qs;
            % add response weight to Q-value
            if cond == 0
                % bandit task => bias toward either arm
                q(itrl) = q(itrl)+br_;
            elseif trlnum(itrl) > 1
                % fairy task => bias toward previous response
                q(itrl) = q(itrl)+br_*(3-2*rprv(itrl));
            end
            % compute response probability
            p(itrl) = 1./(1+exp(-q(itrl)));
            if resp(itrl) == 2
                p(itrl) = 1-p(itrl);
            end
        end
        % compute log-posterior
        ll = sum(log(peps+(1-peps*2)*p)); % log-likelihood
        if ~isinf(sigma)
            l0 = sum(normllh(x,0,sigma)); % log-prior
            lp = ll+l0; % log-posterior
        else
            lp = ll;
        end
    end

end