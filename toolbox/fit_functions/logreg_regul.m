function [b,e] = logreg_regul(x,y,varargin)
%  LOGREG_REGUL  Regularized logistic regression
%
%  Usage: [b,e] = LOGREG_REGUL(x,y,...)
%
%  where x is the design matrix (data points x regressors)
%        y is the dependent variable
%
%  b corresponds to the vector of best-fitting parameter estimates for each
%  regressor. e corresponds to the best-fitting lapse rate (or zero if the
%  optional input argument fit_e is not set to true).
%
%  The function can take the following optional input arguments as name-value
%  pairs:
%    * fit_e   = fit lapse rate? (default = false)
%    * w_regul = weighted regularization? (default = false)
%    * nrun    = number of random starting points (default = 10)
%    * b_sigma = spread of prior on parameter estimates b ~ normpdf(0,b_sigma)
%                (default = [] meaning no prior, should be > 0)
%    * e_betab = prior on lapse rate e ~ betapdf(1,e_betab)
%                (default = [] meaning no prior, should be > 1)
%    * flink   = link function for logistic regression (default = 'logit')
%
%  Example with size(x) = [ndat,2]:
%    [b,e] = logreg_regul(x,y,'fit_e',true,'b_sigma',[2;2],'e_betab',19);
%
%  Note: the prior on the lapse rate controls simultaneously the prior mean and
%  the prior spread. e_betab = 1 means a flat (uniform) prior on the lapse rate.
%  e_betab > 1 means that the prior mean is 1/(1+e_betab) but the prior mode is
%  zero for any value of e_betab > 1 (which makes the prior spread shrink as
%  this value increases). In the above example, e_betab = 19 means that the
%  prior mean for the lapse rate is 5%.
%
%  Valentin Wyart <valentin.wyart@inserm.fr>

% check number of input arguments
narginchk(2,inf);

% get size of design matrix
ndat = size(x,1); % number of data points
nreg = size(x,2); % number of regressors

% check size of dependent variable
assert(ndims(y) == 2 && all(size(y) == [ndat,1]));

% parse optional input arguments
ip = inputParser;
ip.StructExpand = true;  % structure expansion
ip.KeepUnmatched = true; % keep unmatched arguments
ip.addParameter('fit_e',false,@(x)islogical(x) && isscalar(x));
ip.addParameter('w_regul',false,@(x)islogical(x) && isscalar(x));
ip.addParameter('nrun',10,@(x)isnumeric(x) && isscalar(x));
ip.addParameter('b_sigma',[],@(x)isnumeric(x) && ismember(numel(x),[1,nreg]) && all(x > 0));
ip.addParameter('e_betab',[],@(x)isnumeric(x) && isscalar(x) && x >= 1);
ip.addParameter('flink','logit',@(x)ischar(x) && ismember(x,{'logit','probit'}));
ip.parse(varargin{:});

% get optional input arguments
fit_e   = ip.Results.fit_e;      % fit lapse rate?
w_regul = ip.Results.w_regul;    % weighted regularization?
nrun    = ip.Results.nrun;       % number of runs
b_sigma = ip.Results.b_sigma(:); % b ~ norm(0,b_sigma)
e_betab = ip.Results.e_betab;    % e ~ beta(1,e_betab)
flink   = ip.Results.flink;      % link function

if w_regul
    % weighted regularization
    w = mean(x ~= 0,1)';
else
    % flat regularization
    w = ones(nreg,1);
end

% set parameter ranges
b_min = -10+zeros(nreg,1);
b_max = +10+zeros(nreg,1);
e_min = 0;
e_max = 1;
if fit_e
    p_min = [b_min;e_min];
    p_max = [b_max;e_max];
else
    p_min = b_min;
    p_max = b_max;
end

% set fitting options
options = optimset(optimset('fmincon'), ...
    'Display','notify', ...
    'FunValCheck','on', ...
    'Algorithm','interior-point', ...
    'TolX',1e-20,'MaxFunEvals',1e6);

% fit logistic regression model with regularization
pval = nan(nrun,nreg+(fit_e == true));
fval = nan(nrun,1);
for irun = 1:nrun
    % set random starting point
    if ~isempty(b_sigma)
        b_ini = normrnd(zeros(nreg,1),b_sigma);
    else
        b_ini = unifrnd(-10,+10,[nreg,1]);
    end
    if fit_e
        if ~isempty(e_betab)
            e_ini = betarnd(1,e_betab);
        else
            e_ini = unifrnd(0,1);
        end
        p_ini = [b_ini;e_ini];
    else
        p_ini = b_ini;
    end
    [pval(irun,:),fval(irun)] = ...
        fmincon(@(pv)-get_ll(pv),p_ini,[],[],[],[],p_min,p_max,[],options);
end

% get best fit
[~,irun] = min(fval);
b = pval(irun,1:nreg);
if fit_e
    e = pval(irun,1+nreg);
else
    e = 0;
end

    function [ll] = get_ll(p)
        % get logistic regression model log-likelihood
        b_tmp = p(1:nreg);
        dv = x*b_tmp;
        switch flink
            case 'logit'
                pl = 1./(1+exp(-dv));
            case 'probit'
                pl = normcdf(dv);
        end
        if fit_e
            e_tmp = p(1+nreg);
            pl = e_tmp+(1-e_tmp)*pl;
        end
        pl(y == 0) = 1-pl(y == 0);
        ll = sum(log(max(pl,realmin)));
        if ~isempty(b_sigma)
            ll = ll+sum(log(max(normpdf(b_tmp,0,b_sigma),realmin)).*w);
        end
        if fit_e && ~isempty(e_betab)
            ll = ll+log(max(betapdf(e_tmp,1,e_betab),realmin));
        end
    end

end