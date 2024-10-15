%  Primary analyses of (insert paper name here)

%  NOTE: preprocess_data.m should be run first if the /processed/ folder is
%  empty

%% Required code blocks will be labeled (REQ'D)
% clear workspace, figures and command window
clear all
close all
clc

samp    = 'sample2';
condstr = {'bandit','fairy'};
teststr = {'test', 'retest'};

%% Basic behavioral analyses (REQ'D)

% load behavior and ICAR
load(sprintf('./constants/constants_rlinf_%s.mat',samp));           % load nblk, ncond, ntrl, samplename
load(sprintf('./processed/%s/preprocessed_data_%s.mat',samp,samp)); % load the raw data structure for data sample
load(sprintf('./processed/%s/icar_%s.mat',samp,samp));              % load the ICAR scores for data sample
load(sprintf('./processed/%s/idx_excl_ques.mat',samp));             % load subj indices failing attention check
load(sprintf('./processed/age_sex_sample2.mat'));                   % load age/sex into 'demotable'

nsubj = size(idx_fb,1);

% get participants who completed retest and icar
idx_subj = ~isnan(idx_fb(:,1));
idx_icar = ~isnan(icar);
idx_icar(idx_excl_ques) = 0;
fprintf('Number of subjects with complete responses is %d\n', sum(idx_subj));
fprintf('Number of subjects with complete responses and ICAR is %d\n', sum(idx_subj & idx_icar));

% load behavior and ICAR in retest
dat_rt = load(sprintf('./processed/%s_retest/preprocessed_data_%s_retest.mat',samp,samp)); % load the raw data structure for data sample
dat_rt.icar = load(sprintf('./processed/%s_retest/icar_%s_retest.mat',samp,samp)); % load the ICAR scores for the retest sample
dat_rt.icar = dat_rt.icar.icar;
dat_rt.idx_excl_ques = load(sprintf('./processed/%s_retest/idx_excl_ques_retest.mat',samp)); % load subj indices failing attention check
dat_rt.idx_excl_ques = dat_rt.idx_excl_ques.idx_excl_ques;

% get retest participants who completed retest and icar
idx_subj_rt = ~isnan(dat_rt.idx_fb(:,1));
idx_icar_rt = ~isnan(dat_rt.icar);
idx_icar_rt(dat_rt.idx_excl_ques) = 0;
fprintf('Number of retest subjects with complete responses is %d\n', sum(idx_subj_rt));
fprintf('Number of retest subjects with complete responses and ICAR is %d\n', sum(idx_subj_rt & idx_icar_rt));

% create index matrix for subjects who failed attention check during questionnaires (ICAR)
idx_ques_fail = zeros(nsubj,2);
idx_ques_fail(idx_excl_ques,1) = 1;
idx_ques_fail(dat_rt.idx_excl_ques,2) = 1;

% combine test and retest data in one data structure
icar            = [icar     dat_rt.icar];
idx_icar_concat = [idx_icar idx_icar_rt];
idx_subj_concat = [idx_subj idx_subj_rt];


%% Overall accuracy and repetitions (REQ'D)
pcorr = nan(nsubj,2);
prepe = nan(nsubj,2);
pcorr_rt = nan(nsubj,2);
prepe_rt = nan(nsubj,2);

for isubj = 1:nsubj
    if ~idx_subj(isubj)
        continue
    end

    if idx_subj_rt(isubj)
        is_rt_dat = true;
    else
        is_rt_dat = false;
    end
    
    for icond = 0:1
    % Calculating accuracy: ignore the 1st trial of each block
        % match idx_blmn with idx_bmstate rather than idx_corr
        idx_ignore = [1 2 74 75 147 148 220 221]; % condition alignment indices
        idx_trial = setdiff(1:292,idx_ignore);
        resps = idx_blmn(isubj,idx_cond(isubj,:) == icond);
        bmstate = idx_bmstate(isubj,idx_cond(isubj,:) == icond);
        if icond == 0       % bandit: match blmn from t+1 with bmstate of t
            pcorr(isubj,icond+1) = mean(resps(idx_trial) == bmstate(idx_trial-1));
        elseif icond == 1   % fairy:  match blmn from t with bmstate of t 
            pcorr(isubj,icond+1) = mean(resps(idx_trial) == bmstate(idx_trial));
        end
       
        % calculate accuracy in retest
        if is_rt_dat
            resps = dat_rt.idx_blmn(isubj,dat_rt.idx_cond(isubj,:) == icond);
            bmstate = dat_rt.idx_bmstate(isubj,dat_rt.idx_cond(isubj,:) == icond);
            if icond == 0
                pcorr_rt(isubj,icond+1) = mean(resps(idx_trial) == bmstate(idx_trial-1));
            elseif icond == 1
                pcorr_rt(isubj,icond+1) = mean(resps(idx_trial) == bmstate(idx_trial));
            end
        end

    % Calculate repetition rates
        % repetitions must be calculated within each block
        nrepe = 0;
        if is_rt_dat; nrepe_rt = 0; end
        for iblk = 1:4
            idx = idx_cond(isubj,:) == icond & idx_blk(isubj,:) == iblk;
            resps = idx_blmn(isubj,idx);
            if is_rt_dat
                idx = dat_rt.idx_cond(isubj,:) == icond & dat_rt.idx_blk(isubj,:) == iblk;
                resps_rt = dat_rt.idx_blmn(isubj,idx);
            end
            % calculate repeats
            if icond == 0 % bandit (compare starting 2nd to 1st trial)
                nrepe    = nrepe + sum(resps(2:end) == resps(1:end-1));
                if is_rt_dat; nrepe_rt = nrepe_rt  + sum(resps_rt(2:end) == resps_rt(1:end-1)); end
            else % apples: (compare starting 3rd to 2nd trial, since first trial is nothing)
                nrepe    = nrepe + sum(resps(3:end) == resps(2:end-1));
                if is_rt_dat; nrepe_rt = nrepe_rt + sum(resps_rt(3:end) == resps_rt(2:end-1)); end
            end
        end
        
        % calculate total repetition rates
        if icond == 0
            ntot = 72;
        else
            ntot = 71;
        end 
        prepe(isubj,icond+1) = nrepe/(ntot*4);
        if is_rt_dat; prepe_rt(isubj,icond+1) = nrepe_rt/(ntot*4); end
    end
end

% check for binomial test threshold pass
idx_binomialTestPass = pcorr > (158/288); % 158 is the threshold number of correct resposnes to pass a binomial test for randomness given 288 resposnes
idx_binomialTestPass = all(idx_binomialTestPass,2);
idx_binomialTestPass_rt = pcorr_rt > (158/288);
idx_binomialTestPass_rt = all(idx_binomialTestPass_rt,2);
idx_binoTestPass_trt = [idx_binomialTestPass idx_binomialTestPass_rt];
fprintf('Subjects better than chance in test   (%d / %d)\n',sum(idx_binomialTestPass),sum(idx_subj));
fprintf('Subjects better than chance in retest (%d / %d)\n',sum(idx_binomialTestPass_rt),sum(idx_subj_rt));
clearvars idx_binomialTestPass idx_binomialTestPass_rt idx_icar idx_icar_rt idx_subj idx_subj_rt


%% Model-based Analyses of RLINF (REQ'D)
%
%  This script loads fits from merged file, compares model simulations to human
%  data, and plots results in terms of reversal/switch curves, psychometric
%  curves and logistic regression kernels.
%
%  Check the documentation of each code cell for more information.
%
%  Jun Seok Lee <jun.seok.lee@ens.fr>
%  Valentin Wyart <valentin.wyart@inserm.fr>

addpath('./toolbox/stat_functions/');
addpath('./toolbox/fit_functions/');
addpath('./toolbox/plot_functions/');

% set parameters
pathname = './fits/merged';         % merged fits path name
modname  = 'fit_model_inf_dist';    % model name
sfit     = '10111';                 % fitting configuration string (model-specific)
nullp    = 0.05;                    % p-value threshold for random strategy subjects

fprintf('P-value threshold for null strategy is %.3f\n',nullp);

out_vbmc_trt = cell(2,1);
subjlist_trt = cell(2,1);
is_miss_trt = zeros(nsubj,2,2); % subj, cond, test/retest
is_stim_trt = zeros(nsubj,2,2);
is_hinf_trt = zeros(nsubj,2,2);
is_null_trt = zeros(nsubj,2,2);

fprintf('Loading fits...\n');
for itest = 1:2
    if itest == 1
        samp_ana = samp;
    else
        samp_ana = sprintf('%s_retest',samp);
    end
    fprintf('Analyzing data from %s...\n',teststr{itest});
    
    % load merged datafile
    fname = sprintf('%s_%s_%s.mat',modname,samp_ana,sfit);
    fname = fullfile(pathname,fname);
    if ~exist(fname,'file')
        error('Merged datafile not found!');
    end
    load(fname,'condlist','out_vbmc');
    out_vbmc_trt{itest} = out_vbmc;
    
    % set subject list as subjects with valid fits in both conditions
    subjlist = find(all(~cellfun(@isempty,out_vbmc),2));
    subjlist_trt{itest} = subjlist;
    
    % label subjects in terms of strategy  
%     [is_miss,is_stim,is_hinf,is_null] = get_fit_stim_hinf(samp_ana,subjlist);
    [is_miss,is_stim,is_hinf,is_null] = get_fit_stim_hinf(samp_ana,subjlist,'pnul_typ','acc','pnul_thr',nullp);

    is_miss_trt(subjlist,:,itest)     = is_miss;
    is_stim_trt(subjlist,:,itest)     = is_stim;
    is_hinf_trt(subjlist,:,itest)     = is_hinf;
    is_null_trt(subjlist,:,itest)     = is_null;

    nall = sum(idx_subj_concat(:,itest));
    for icond = 1:2
        n_null = sum(is_null(:,icond));
        n_stim = sum(is_stim(:,icond));
        n_hinf = sum(is_hinf(:,icond));

        fprintf('Number of subjects best fit by null/random model in %s (%d / %d) %.3f\n', condstr{icond}, n_null, nall, n_null/nall);
        fprintf('Number of subjects best fit by stimulus model in %s (%d / %d) %.3f\n', condstr{icond}, n_stim, nall, n_stim/nall);
        fprintf('Number of subjects best fit by inference model in %s (%d / %d) %.3f\n', condstr{icond}, n_hinf, nall, n_hinf/nall);
    end

    fprintf('Number of subjects best fit by inference model in %s (%d / %d)\n\n', teststr{itest}, sum(all(is_hinf,2)), sum(idx_subj_concat(:,itest)));
end
fprintf('Done.\n')

% Subject inclusion indices for test
idx_incl_tt = all(is_hinf_trt(:,:,1),2);
fprintf('Number of subjects in inclusion indices for test (%d)\n', sum(idx_incl_tt));
% Subject inclusion indices for retest
idx_incl_rt = idx_incl_tt & idx_subj_concat(:,2);
fprintf('Number of subjects in inclusion indices for retest (%d)\n', sum(idx_incl_rt));

% combined inclusion indices for test and retest
idx_incl_trt = [idx_incl_tt idx_incl_rt];

%% Run behavioral analyses on data and model simulations (REQ'D)

% clear figures and command window
close all
clc

% User input: Skip logistic regressions? (these take a while to process)
skip_regression = true; 

% User input: Skip everything except parameter logging
skip_everything = false; 

% set parameters
fittype = 'avg'; % best-fitting parameter type (avg/map)
nbreg    = 10;    % number of repetitions for simulated logistic regressions
nbin     = 10;    % number of stimulus weight bins
bsigma   = 3;     % regularization prior s.d. for stimulus weights
nlag     = 5;     % maximum lag for logistic regression analysis
h_opt    = 0.116; % optimal hazard rate

% initialize structure with best-fitting parameter values
out = out_vbmc_trt{1}{1,1};
pfit = [];
pfit.func = modname;
pfit.type = fittype;
pfit.free = out.xnam;
pfit.fixd = setdiff(out.pnam,out.xnam);
for ipar = 1:numel(out.pnam)
    pfit.(out.pnam{ipar}) = nan(nsubj,2,2);
end
% parameters
parstr  = out.pnam;
npar    = numel(out.xnam);
pars    = nan(nsubj,npar,2,2); % subj, par, cond, session

% overall accuracy and switch rate
pcor_sub_all  = nan(nsubj,2,2); % subj, cond, session
pnul_sub_all  = nan(nsubj,2,2);
pswi_sub_all  = nan(nsubj,2,2);
pcor_sim_all  = nan(nsubj,2,2);
pswi_sim_all  = nan(nsubj,2,2);
% reversal and switch curves
prevt_sub_all = nan(nsubj,10,2,2); % subj, position, cond, session
pswit_sub_all = nan(nsubj,10,2,2);
prevt_sim_all = nan(nsubj,10,2,2);
pswit_sim_all = nan(nsubj,10,2,2);
% fraction repeat wrt evidence direction
ppsy_sub_all  = nan(nsubj,10,2,2);
ppsy_sim_all  = nan(nsubj,10,2,2);
% logistic regression weights
breg_sub_all  = nan(nsubj,nlag+1,2,2);
breg_sim_all  = nan(nsubj,nlag+1,2,2);
% stimulus weights
bs_sub_all    = nan(nsubj,nbin,2,2);
bs_sim_all    = nan(nsubj,nbin,2,2);
% response weights
br_sim_all    = nan(nsubj,2,2);
br_sub_all    = nan(nsubj,2,2);
 
% for saving
out_sim_save = cell(247,2,2); % subj, cond, session
out_opt_save = cell(247,2,2);

for itest = 1:2
    hbar = waitbar(0,'');
    fprintf('Processing data from %s...\n', teststr{itest});

    out_vbmc = out_vbmc_trt{itest};
    % indices of subjects to compute data on
    subjlist = find(idx_subj_concat(:,itest)==1);
    if itest == 1
        samp_ana = samp;
    else
        samp_ana = sprintf('%s_retest',samp);
    end
    
    for isubj = 1:nsubj
        if ~ismember(isubj,subjlist)
            continue
        end
        waitbar(isubj/nsubj,hbar,sprintf('processing S%03d',isubj));
        for icond = 1:2
            % get subject and condition of interest
            subj = isubj;
            cond = condlist(icond);
    
            % get fitting output
            out = out_vbmc{isubj,icond};
    
            % get best-fitting parameter values
            switch fittype
                case 'avg' % use posterior average
                    out = use_xavg(out);
                case 'map' % use posterior maximum
                    out = use_xmap(out);
                otherwise
                    error('Undefined best-fitting parameter type!');
            end
            for ipar = 1:numel(out.pnam)
                pfit.(out.pnam{ipar})(isubj,icond,itest) = out.(out.pnam{ipar});
            end
            pars(isubj,:,icond,itest) = out.(sprintf('x%s',fittype));
            if skip_everything
                continue
            end
            
            % get data structure
            dat = dat_model_inf(samp_ana,subj,cond);
            trlnum  = dat.trlnum;  % trial number in current block
            bmstate = dat.bmstate; % hidden state direction
            efac    = dat.efac;    % evidence scaling factor
            stim    = dat.stim;    % stimulus direction
            evid    = stim*efac;   % evidence direction
    
            % get number of trials
            ntrl = numel(trlnum);
    
            % simulate best-fitting model
            funfit = str2func(modname);
            if strcmpi(sfit,'11111')
                out = rmfield(out,'fittype');
            end
            out_sim = funfit(dat,out);
    
            % simulate optimal model
            out_opt = funfit(dat,out,'nsmp',1, ...
                'h',h_opt,'gamma',0,'alpha',0','omega',0,'sigma',0,'eta',0','tau',0);
                % for saving
                out_opt_save{subj,icond,itest} = out_opt;
                out_opt_save{subj,icond,itest}.other = struct;
            
            % compute psychometric curve (Bayes-optimal evidence)
            xbin = discretize(out_opt.xt,[-inf,(-4:+4)/4*2,+inf]);
            for ibin = 1:nbin
                itrl = xbin == ibin;
                pevi_opt_all(isubj,ibin,icond,itest) = mean(out_opt.rs(itrl) == 1);
                pevi_sub_all(isubj,ibin,icond,itest) = mean(dat.resp(itrl) == 1);
                pevi_sim_all(isubj,ibin,icond,itest) = mean(out_sim.rs(itrl,:) == 1,'all');
            end
    
            % compute overall statistics
            itrl = find(trlnum > 1);
    
            pcat_tru_all(isubj,icond,itest) = mean(bmstate(itrl) == 1); 
            % 0/ for optimal model
            pcat_opt_all(isubj,icond,itest) = mean(out_opt.rs(itrl) == 1);
            pcor_opt_all(isubj,icond,itest) = mean(out_opt.rs(itrl) == bmstate(itrl));
            pswi_opt_all(isubj,icond,itest) = 1-mean(out_opt.rs(itrl) == out_opt.rs(itrl-1));
                out_opt_save{subj,icond,itest}.other.pcat = pcat_opt_all(isubj,icond,itest);
                out_opt_save{subj,icond,itest}.other.pcor = pcor_opt_all(isubj,icond,itest);
                out_opt_save{subj,icond,itest}.other.pswi = pswi_opt_all(isubj,icond,itest);
            % 1/ for human data
            pcat_sub_all(isubj,icond,itest) = mean(dat.resp(itrl) == 1);
            pcor_sub_all(isubj,icond,itest) = mean(dat.resp(itrl) == bmstate(itrl));
            pnul_sub_all(isubj,icond,itest) = binocdf(nnz(dat.resp(itrl) == bmstate(itrl)),numel(itrl),0.5,'upper');
            pswi_sub_all(isubj,icond,itest) = 1-mean(dat.resp(itrl) == dat.resp(itrl-1));        
            
            % 2/ for simulations of best-fitting model
            pcat_sim_all(isubj,icond,itest) = mean(out_sim.rs(itrl,:) == 1,'all');
            pcor_sim_all(isubj,icond,itest) = mean(bsxfun(@eq,out_sim.rs(itrl,:),bmstate(itrl)),'all');
            pswi_sim_all(isubj,icond,itest) = 1-mean(out_sim.rs(itrl,:) == out_sim.rs(itrl-1,:),'all');
                out_sim_save{subj,icond,itest}.other.pcat = pcat_sim_all(isubj,icond,itest);
                out_sim_save{subj,icond,itest}.other.pcor = pcor_sim_all(isubj,icond,itest);
                out_sim_save{subj,icond,itest}.other.pswi = pswi_sim_all(isubj,icond,itest);
    
            % compute reversal and switch curves
            isrev = [false;bmstate(2:end) ~= bmstate(1:end-1)];
            isrev(trlnum == 1) = false;
            irev = find(isrev);
            nrev = numel(irev);
            % 0/ for optimal model
            rrev = [];
            rrep = [];
            for i = 1:nrev
                j = irev(i)+(-5:+4);
                rrev(i,:) = out_opt.rs(j) == bmstate(irev(i));
                rrep(i,:) = out_opt.rs(j) == out_opt.rs(j-1);
            end
            prevt_opt_all(isubj,:,icond,itest) = mean(rrev,1);
            pswit_opt_all(isubj,:,icond,itest) = 1-mean(rrep,1);
                out_opt_save{subj,icond,itest}.other.prevt = prevt_opt_all(isubj,:,icond,itest);
                out_opt_save{subj,icond,itest}.other.pswit = pswit_opt_all(isubj,:,icond,itest);
            % 1/ for human data
            rrev = [];
            rrep = [];
            for i = 1:nrev
                j = irev(i)+(-5:+4);
                rrev(i,:) = dat.resp(j) == bmstate(irev(i));
                rrep(i,:) = dat.resp(j) == dat.resp(j-1);
            end
            prevt_sub_all(isubj,:,icond,itest) = mean(rrev,1);
            pswit_sub_all(isubj,:,icond,itest) = 1-mean(rrep,1);
            % 2/ for simulations of best-fitting model
            rrev = [];
            rrep = [];
            for i = 1:nrev
                j = irev(i)+(-5:+4);
                rrev(i,:,:) = out_sim.rs(j,:) == bmstate(irev(i));
                rrep(i,:,:) = out_sim.rs(j,:) == out_sim.rs(j-1,:);
            end
            prevt_sim_all(isubj,:,icond,itest) = mean(rrev,[1,3]);
            pswit_sim_all(isubj,:,icond,itest) = 1-mean(rrep,[1,3]);
                out_sim_save{subj,icond,itest}.other.prevt = prevt_sim_all(isubj,:,icond,itest);
                out_sim_save{subj,icond,itest}.other.pswit = pswit_sim_all(isubj,:,icond,itest);
    
            % compute fraction repeat wrt evidence direction
            itrl = find(trlnum > 1);
    
            % 0/ for optimal model
            ppsy_opt = nan(10,1);
            xreg = evid(itrl).*(3-2*out_opt.rs(itrl-1));
            yreg = out_opt.rs(itrl) == out_opt.rs(itrl-1);
            ibin = discretize(xreg,linspace(-efac,+efac,11));
            for i = 1:10
                ppsy_opt(i) = mean(yreg(ibin == i));
            end
            ppsy_opt_all(isubj,:,icond,itest) = ppsy_opt;
                out_opt_save{subj,icond,itest}.other.prepevi = ppsy_opt_all(isubj,:,icond,itest);
            % 1/ for human data
            ppsy_sub = nan(10,1);
    
            xreg = evid(itrl).*(3-2*dat.resp(itrl-1));
            yreg = dat.resp(itrl) == dat.resp(itrl-1);
            ibin = discretize(xreg,linspace(-efac,+efac,11));
            
            for i = 1:10 % number of bins
                ppsy_sub(i) = mean(yreg(ibin == i));
            end
            ppsy_sub_all(isubj,:,icond,itest) = ppsy_sub;
    
            % 2/ for simulations of best-fitting model
            ppsy_sim = nan(10,out.nsmp);
            for ismp = 1:out.nsmp
                xreg = evid(itrl).*(3-2*out_sim.rs(itrl-1,ismp));
                yreg = out_sim.rs(itrl,ismp) == out_sim.rs(itrl-1,ismp);
                ibin = discretize(xreg,linspace(-efac,+efac,11));
                for i = 1:10
                    ppsy_sim(i,ismp) = mean(yreg(ibin == i));
                end
            end
            ppsy_sim = mean(ppsy_sim,2);
            ppsy_sim_all(isubj,:,icond,itest) = mean(ppsy_sim,2);
                out_sim_save{subj,icond,itest}.other.prepevi = ppsy_sim_all(isubj,:,icond,itest);
            
            if skip_regression 
                continue
            end
    
            % run logistic regression analysis
            itrl = find(trlnum > nlag);
            xreg = evid(itrl);
            for ireg = 1:nlag
                xreg = cat(2,xreg,evid(itrl-ireg));
            end
            % 1/ for human data
            breg = glmfit(xreg,dat.resp(itrl) == 1,'binomial','link','logit');
            breg_sub_all(isubj,:,icond,itest) = breg(2:end);
            bint_sub_all(isubj,icond) = breg(1);
            % 2/ for simulations of best-fitting model
            if nbreg == 0
                % stack all data together
                xreg = repmat(xreg,[out.nsmp,1]);
                breg = glmfit(xreg,reshape(out_sim.rs(itrl,:),[],1) == 1,'binomial','link','logit');
                breg_sim_all(isubj,:,icond,itest) = breg(2:end);
            else
                breg_sim = nan(nlag+1,out.nsmp);
                for ismp = 1:nbreg
                    breg = glmfit(xreg,out_sim.rs(itrl,ismp) == 1,'binomial','link','logit');
                    breg_sim(:,ismp) = breg(2:end);
                end
                breg_sim_all(isubj,:,icond,itest) = nanmedian(breg_sim,2);
            end
            
            % fit stimulus weights
            [bs,br] = fit_bs(dat,'nbin',nbin,'sigma',bsigma);
            bs_sub_all(isubj,:,icond,itest) = bs;
            br_sub_all(isubj,icond,itest) = br;
            if nbreg == 0
                % stack all data together
                dat_sim = dat;
                fnames = fieldnames(dat_sim);
                for i = 1:numel(fnames)
                    if size(dat_sim.(fnames{i}),1) == ntrl
                        dat_sim.(fnames{i}) = repmat(dat_sim.(fnames{i}),[out.nsmp,1]);
                    end
                end
                resp_sim = out_sim.rs;
                rprv_sim = cat(1,nan(1,out.nsmp),resp_sim(1:end-1,:));
                rprv_sim(trlnum == 1,:) = nan;
                dat_sim.resp = resp_sim;
                dat_sim.rprv = rprv_sim;
                [bs,br] = fit_bs(dat_sim,'nbin',nbin,'sigma',bsigma);
                bs_sim_all(isubj,:,icond,itest) = bs;
                br_sim_all(isubj,icond,itest) = br;
            else
                dat_sim = dat;
                bs_sim = nan(nbin,out.nsmp);
                br_sim = nan(1,out.nsmp);
                for ismp = 1:nbreg
                    resp_sim = out_sim.rs(:,ismp);
                    rprv_sim = cat(1,nan,resp_sim(1:end-1));
                    rprv_sim(trlnum == 1) = nan;
                    dat_sim.resp = resp_sim;
                    dat_sim.rprv = rprv_sim;
                    [bs,br] = fit_bs(dat_sim,'nbin',nbin,'sigma',bsigma);
                    bs_sim(:,ismp) = bs;
                    br_sim(ismp) = br;
                end
                bs_sim_all(isubj,:,icond,itest) = nanmedian(bs_sim,2);
                br_sim_all(isubj,icond,itest) = nanmedian(br_sim);
            end
        end
    end
    close(hbar);
end
fprintf('Done.\n');

% Log-transform parameters based on their bounds
npar = size(pars,2);
pars_logt = pars;
for ipar = 1:npar
    % reshape distributions to be approx. normal
    if ipar == 1 % bounded [0 1]
        pars_logt(:,ipar,:,:) = log(pars_logt(:,ipar,:,:)./(1-pars_logt(:,ipar,:,:)));
    elseif ismember(ipar, [3 5]) % bounded [0 Inf)
        pars_logt(:,ipar,:,:) = log(pars_logt(:,ipar,:,:));
    end
end


%% Note on following analyses
%  By default, the inclusion indices are as such:
%
%  Model-free analyses:
%   idx_subj_concat:      all subjects w/ behavioral data
%   idx_binoTestPass_trt: subjects who passed random response test
% 
%  Model-based analyses:
%   is_hinf_trt: subjects best fit by hidden-state inference model
%   is_stim_trt: subjects best fit by stimulus model


%% Plot summary performance (optimal, subject, model fit simulation)

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session = 'test';   % set 'session' to either 'test' or 'retest'
src     = 'sub';    % 'opt', 'sub', 'sim' (optimal, subject, sim)
dtype   = 'cor';    % 'cor','swi' (pcorrect, pswitch)
is_savetofile  = true; % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
isok = idx_incl_trt(:,itest);
nok = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);
% figure parameters
figh = 4; % figure height (cm)

% plotting parameters
datwid = 0.2;
mrksz  = 10;
c = [.25 .56 .58; .9 .47 .18]; %linspecer(2);

xsrc = eval(sprintf('p%s_%s_all',dtype,src));
[p,h,stats] = signrank(xsrc(isok,1,itest),xsrc(isok,2,itest));
fprintf('p%s (bandit): %.3f±%.3f\n',dtype,mean(xsrc(isok,1,itest)),std(xsrc(isok,1,itest)));
fprintf('p%s (fairy): %.3f±%.3f\n',dtype,mean(xsrc(isok,2,itest)),std(xsrc(isok,2,itest)));
fprintf('Comparing location differences between bandit and fairy for p%s...\n',dtype);
fprintf('p=%.3f, Z=%.3f\n',p,stats.zval);

hf = figure('Color','white');
clf
hold on
ncond = 2;
if strcmpi(src,'opt')
    ncond = 1;
    condstr = {'bandit / fairy'};
    c = [1 1 1]*.8;
end
for icond = 1:ncond
    % plot scattered violin 
    x = xsrc(isok,icond,itest);
    xk = linspace(min(x),max(x),100);
    pk = ksdensity(x,xk);
    pk = pk/max(pk);
    str = interp1(xk,pk,x);
    jit = linspace(-1,+1,numel(x));
    xpos = nan(size(x));
    for j = randperm(numel(x))
        xpos(j) = icond+jit(j)*str(j)*datwid;
    end
    % connecting lines
    if ~strcmpi(src,'opt')
        if icond == 1
            xloc = xpos;
            yloc = x;
        else
            xloc = [xloc, xpos];
            yloc = [yloc, x];
        end
        if icond == ncond
            plot(xloc',yloc','-','LineWidth',0.25,'Color',[1 1 1 .5]*.8);
        end
    end
    % scattered violin
    p = scatter(xpos,x,mrksz*.5, ...
        'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','none');
    p.MarkerFaceAlpha = 0.5;

    plot([1 1]*icond,std(x)/sqrt(sum(isok)),'Color',c(icond,:),'LineWidth',1); % error bars (here SEM) for optimal
    scatter(icond,mean(x),mrksz*2,[1 1 1],'LineWidth',1,'MarkerFaceColor',c(icond,:)); % means
end

xticks(1:ncond);
% xticklabels(condstr(1:ncond));
xlim([.5 ncond+.5]);
switch dtype
    case 'cor'
        yticks(.6:.1:.9);
        ylim([.55 .95]);
        ylblstr = 'correct';
    case 'swi'
        yticks(.1:.1:.4);
        ylim([.05 .45]);
        ylblstr = 'switch';
end

pbar = 1/2*(3+0.2)/2.2; % plot box aspect ratio

set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
xlabel('condition','FontSize',8);
ylabel(sprintf('fraction %s',ylblstr),'FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes)
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end

if is_savetofile
    % save figure to file
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = sprintf('./figs/p%s_%s_avg_%s',dtype,src,teststr{itest});
    print(fname,'-painters','-dpdf');
end

% between-condition scatter plot
if strcmpi(src,'opt')
    return
end
hf = figure('Color','white');
clf
hold on

xdots = xsrc(isok,1,itest);
ydots = xsrc(isok,2,itest);
scatter(xdots,ydots,mrksz*.5,'MarkerFaceColor',[1 1 1]*.8, ...
        'MarkerFaceAlpha',.8,'MarkerEdgeColor','none');
xrange = min(xdots):.01:max(xdots);
[pn,s] = polyfit(xdots,ydots,1);
[py,d] = polyconf(pn,xrange,s,'alpha',0.05,'predopt','curve');
s = shadedErrorBar(xrange,py,d,'patchSaturation',.1,'lineprops',{'LineWidth',1,'Color',[1 1 1]*.8});
set(s.edge,'LineStyle','none');
s.HandleVisibility = 'off';
[r,p] = corr(xdots,ydots);
fprintf('Corr: r=%+.4f, p=%.4f\n',r,p);

switch dtype
    case 'cor'
        yticks(.6:.1:.9);
        xticks(.6:.1:.9);
        lims = [.55 .95];
    case 'swi'
        yticks(.1:.1:.4);
        xticks(.1:.1:.4);
        lims = [.05 .45];
end
xlim(lims);
ylim(lims);
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
xlabel('bandit','FontSize',8);
ylabel('fairy','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes)
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end

if is_savetofile
    % save figure to file
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = sprintf('./figs/corr_p%s_%s_avg_%s',dtype,src,teststr{itest});
    print(fname,'-painters','-dpdf');
end

% relation to icar
isok    = idx_icar_concat(:,1) & idx_subj_concat(:,1); 
if itest == 2
    isok = isok & idx_icar_concat(:,2) & idx_subj_concat(:,2);
end
nok     = sum(isok);

hf = figure('Color','white');
clf
hold on
for icond = 1:2
    xdots = icar(isok,itest);
    ydots = xsrc(isok,icond,itest);
    scatter(xdots,ydots,mrksz*.5,'MarkerFaceColor',c(icond,:), ...
            'MarkerFaceAlpha',.8,'MarkerEdgeColor','none');
    xrange = min(xdots):.01:max(xdots);
    [pn,s] = polyfit(xdots,ydots,1);
    [py,d] = polyconf(pn,xrange,s,'alpha',0.05,'predopt','curve');
    s = shadedErrorBar(xrange,py,d,'patchSaturation',.1,'lineprops',{'LineWidth',1,'Color',c(icond,:)});
    set(s.edge,'LineStyle','none');
    s.HandleVisibility = 'off';
    [r,p] = corr(xdots,ydots,'Type','Spearman');
    fprintf('Corr: r=%+.4f, p=%.4f\n',r,p);
end
xlim([-.5 16.5])
xticks(0:4:16)
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
xlabel('ICAR-16','FontSize',8);
ylabel(sprintf('p%s',dtype),'FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes)
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end

if is_savetofile
    % save figure to file
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = sprintf('./figs/p%s_%s_icar_%s',dtype,src,teststr{itest});
    print(fname,'-painters','-dpdf');
end


%% Plot reversal curves (p(reversed) p(switched))

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session = 'test';     % set 'session' to either 'test' or 'retest'
is_sim_overlay = true; % set to true to show simulated behavior
is_savetofile  = false; % set to true to save figure to file (in /figs/)

% figure settings (do not change unless resizing of figure desired)
pbar = 4/3; % plot box aspect ratio
figw = 12; % figure width (cm)
figh = 8; % figure height (cm)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
isok = idx_incl_trt(:,itest);
nok = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);

% define condition-specific colors (1=bandit 2=fairy)
c = [.25 .56 .58; .9 .47 .18]; %linspecer(2);
% c = flipud(c);

hf = figure;
% get group-level statistics for reversal curves
pavg_opt = squeeze(mean(prevt_opt_all(isok,:,:,itest),1));
pavg_sub = squeeze(mean(prevt_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(prevt_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(prevt_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(prevt_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;

xpos = (-5:+4)+0.5;
for icond = 1:2
    xdsp = icond*.2-.3; % x displacement for condition display (too much overlap)
    subplot(1,2,1);% for 1x2 tiling 
    hold on
    xlim([-5,+5]);
    ylim([0.1,0.9]);
    if icond == 1
        plot(xpos,pavg_opt(:,icond),'-','LineWidth',1.50,'Color',[1,1,1]*0.8); % optimal
    end
    % sim
    if is_sim_overlay
        patch([xpos,flip(xpos)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xpos,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:)); 
    end
    plot(xlim,[0.5,0.5],'-','Color',0.8*[1,1,1],'LineWidth',0.75); 
    plot([0,0],ylim,'k-','LineWidth',0.75);
    % subjects
    for i = 1:10
        plot(xpos([i,i])+xdsp,pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:)); % error
    end
    if ~is_sim_overlay
        plot(xpos+xdsp,pavg_sub(:,icond),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
    end
    plot(xpos+xdsp,pavg_sub(:,icond),'wo','MarkerSize',4,'MarkerFaceColor',c(icond,:),'LineWidth',0.75); % subject markers
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',[-3.5,-1.5,+1.5,+3.5],'XTickLabel',{'-4','-2','2','4'});
    set(gca,'YTick',0:0.2:1);
    xlabel('position from reversal','FontSize',8);
    ylabel('fraction reversed','FontSize',8);
end

% get group-level statistics for switch curves
pavg_opt = squeeze(mean(pswit_opt_all(isok,:,:,itest),1));
pavg_sub = squeeze(mean(pswit_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(pswit_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(pswit_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(pswit_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;

xpos = (-5:+4)+0.5;
for icond = 1:2
    xdsp = icond*.2-.3; % x displacement for condition display (too much overlap)
    subplot(1,2,2);
    hold on
    xlim([-5,+5]);
    ylim([0.05,0.45]);
    if icond == 1
        plot(xpos,pavg_opt(:,icond),'-','LineWidth',1.50,'Color',[1,1,1]*0.8); % optimal
    end
    % simulation
    if is_sim_overlay
        patch([xpos,flip(xpos)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xpos,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:)); % sim
    end
    plot(xlim,[0.5,0.5],'-','Color',0.8*[1,1,1],'LineWidth',0.75);
    plot([0,0],ylim,'k-','LineWidth',0.75);
    % subject
    if true
        for i = 1:10
            plot(xpos([i,i])+xdsp,pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:));
        end
    end
    if ~is_sim_overlay
        plot(xpos+xdsp,pavg_sub(:,icond),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
    end
    plot(xpos+xdsp,pavg_sub(:,icond),'wo','MarkerSize',4,'MarkerFaceColor',c(icond,:),'LineWidth',0.75); % subject
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',[-3.5,-1.5,+1.5,+3.5],'XTickLabel',{'-4','-2','2','4'});
    set(gca,'YTick',0:0.1:1);
    xlabel('position from reversal','FontSize',8);
    ylabel('fraction switch','FontSize',8);
end

% print to file
if ~exist('./figs','dir')
    mkdir('./figs');
end
figure(hf);
axes = findobj(gcf,'type','axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
set(gcf,'PaperPositionMode','manual', ...
    'PaperPosition',[0.5*(21.0 -figw),0.5*(29.7-figh),figw,figh],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
if is_savetofile
    print(sprintf('./figs/fig_prev_%s_%s_%s',modname,sfit,teststr{itest}),'-painters','-dpdf');
end

%% Plot psychometric curves 

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session = 'test';     % set 'session' to either 'test' or 'retest'
is_sim_overlay = true; % set to true to show simulated behavior
is_savetofile  = true; % set to true to save figure to file (in /figs/)

% figure settings (do not change unless resizing of figure desired)
pbar = 4/3; % plot box aspect ratio
figw = 12; % figure width (cm)
figh = 8; % figure height (cm)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

isok = idx_incl_trt(:,itest);
nok = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
nok = sum(isok);

% define condition-specific colors (1=bandit 2=fairy)
c = [.25 .56 .58; .9 .47 .18]; 

hf = figure;
clf
% get group-level statistics for reversal curves
pavg_opt = squeeze(mean(ppsy_opt_all(isok,:,:,itest),1));
pavg_sub = squeeze(mean(ppsy_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(ppsy_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(ppsy_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(ppsy_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;

xval = linspace(-efac,+efac,11);
xval = 0.5*(xval(1:end-1)+xval(2:end));
for icond = 1:2
    subplot(1,2,1); 
    hold on
    xlim([-1.8,+1.8]);
    ylim([0,1]);
    % model
    if is_sim_overlay
        patch([xval,flip(xval)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xval,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:));
    end
    % optimal
    plot(xval,pavg_opt(:,icond),'-','LineWidth',1.50,'Color',[1,1,1]*0.8);
    plot(xlim,[0.5,0.5],'-','Color',0.8*[1,1,1],'LineWidth',0.75);
    plot([0,0],ylim,'k-','LineWidth',0.75);
    
    % subjects
    for i = 1:10
        plot(xval([i,i]),pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:));
    end
    plot(xval,pavg_sub(:,icond),'o-','Color',c(icond,:),'MarkerSize',4,'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','w','LineWidth',0.75);
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',-1.5:1:+1.5);
    set(gca,'YTick',0:0.2:1);
    xlabel('stimulus direction','FontSize',8);
    ylabel('fraction repeat','FontSize',8);
end

% get group-level statistics for reversal curves
pavg_sub = squeeze(mean(pevi_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(pevi_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(pevi_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(pevi_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_opt = squeeze(mean(pevi_opt_all(isok,:,:,itest),1));

xval = -2.25:0.5:+2.25;
for icond = 1:2
    subplot(1,2,2);
    hold on
    xlim([-1,+1]*2.5);
    ylim([0,1]);
    % model
    if is_sim_overlay
        patch([xval,flip(xval)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xval,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:));
    end
    % optimal
    plot(xval,pavg_opt(:,icond),'-','LineWidth',1.50,'Color',[1,1,1]*0.8);
    plot(xlim,[0.5,0.5],'-','Color',0.8*[1,1,1],'LineWidth',0.75);
    plot([0,0],ylim,'k-','LineWidth',0.75);
    % subjects
    for i = 1:10
        plot(xval([i,i]),pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:));
    end
    plot(xval,pavg_sub(:,icond),'o-','Color',c(icond,:),'MarkerSize',4,'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','w','LineWidth',0.75);
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',-2.0:1:+2.0,'XTickLabel',{'-2.0','-1.0','0','1.0','2.0'});
    set(gca,'YTick',0:0.2:1);
    xlabel('Bayes-optimal evidence','FontSize',8);
    ylabel('fraction option A','FontSize',8);

end

% print to file
if ~exist('./figs','dir')
    mkdir('./figs');
end
figure(hf);
axes = findobj(gcf,'type','axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
set(gcf,'PaperPositionMode','manual', ...
    'PaperPosition',[0.5*(21.0 -figw),0.5*(29.7-figh),figw,figh],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
if is_savetofile
    print(sprintf('./figs/fig_ppsy_%s_%s_%s',modname,sfit,teststr{itest}),'-painters','-dpdf');
end

%% Plot logistic regression results

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session = 'test';     % set 'session' to either 'test' or 'retest'
is_sim_overlay = true; % set to true to show simulated behavior
is_savetofile  = true; % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

isok = idx_incl_trt(:,itest);
nok = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);

% figure settings (do not change unless resizing of figure desired)
pbar = 4/3; % plot box aspect ratio
figw = 12; % figure width (cm)
figh = 8; % figure height (cm)

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
nok = sum(isok);

% define condition-specific colors (1=bandit 2=fairy)
c = [.25 .56 .58; .9 .47 .18]; 

hf = figure;
% regression on value
pavg_sub = squeeze(mean(bs_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(bs_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(bs_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(bs_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;

for icond = 1:2
    subplot(2,2,(icond-1)*2+1);
    hold on
    if icond == 1
        xpos = linspace(0,100,nbin+1);
        xpos = 0.5*(xpos(1:end-1)+xpos(2:end));
        xlim([0,100]);
        ylim([-1,+4]);
    else
        xpos = linspace(-1,+1,nbin+1);
        xpos = 0.5*(xpos(1:end-1)+xpos(2:end));
        xlim([-1,+1]);
        ylim([-3,+3]);
    end
    xdsp = (icond*.2-.3)/4 ; % x displacement for condition display (too much overlap)
    plot(xlim,[0,0],'-','Color',0.8*[1,1,1],'LineWidth',0.75);
    % model
    if is_sim_overlay
        patch([xpos,flip(xpos)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xpos,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:));
    end
    % subjects
    if ~is_sim_overlay
        plot(xpos+xdsp,pavg_sub(:,icond),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
    end
    for i = 1:10
        plot(xpos([i,i]),pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:));
    end
    plot(xpos,pavg_sub(:,icond),'wo','MarkerSize',4,'MarkerFaceColor',c(icond,:),'LineWidth',0.75);
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    if icond == 1
        set(gca,'XTick',[0:25:100]);
        xlabel('reward','FontSize',8);
        ylabel('decision value','FontSize',8);
    else
        set(gca,'XTick',[-1:0.5:+1]);
        xlabel('color','FontSize',8);
        ylabel('decision value','FontSize',8);
    end
    set(gca,'YTick',-2:2:+4,'YTickLabel',{'-2.0','0','2.0','4.0'});
end

% regression on time
pavg_sub = squeeze(mean(breg_sub_all(isok,:,:,itest),1));
pc95_sub = squeeze(std(breg_sub_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;
pavg_sim = squeeze(mean(breg_sim_all(isok,:,:,itest),1));
pc95_sim = squeeze(std(breg_sim_all(isok,:,:,itest),[],1))/sqrt(nok)*1.96;

xpos = 0:5;
for icond = 1:2
    xdsp = (icond*.2-.3)/4 ; % x displacement for condition display (too much overlap)
    subplot(2,2,(icond-1)*2+2);
    hold on
    xlim([-0.3,+5.3]);
    ylim([-0.5,2.5]);
    plot(xlim,[0,0],'-','Color',0.8*[1,1,1],'LineWidth',0.75);
    % model
    if is_sim_overlay
        patch([xpos,flip(xpos)], ...
            [pavg_sim(:,icond)+pc95_sim(:,icond);flip(pavg_sim(:,icond)-pc95_sim(:,icond))]', ...
            0.5*(c(icond,:)+1)-0.01,'EdgeColor','none','FaceAlpha',0.5);
        plot(xpos,pavg_sim(:,icond),'-','LineWidth',1.50,'Color',c(icond,:));
    end
    % subjects
    if ~is_sim_overlay
        plot(xpos+xdsp,pavg_sub(:,icond),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
    end
    for i = 1:6
        plot(xpos([i,i])+xdsp,pavg_sub(i,icond)+pc95_sub(i,icond)*[-1,+1],'-','LineWidth',0.75,'Color',c(icond,:));
    end
    plot(xpos+xdsp,pavg_sub(:,icond),'wo','MarkerSize',4,'MarkerFaceColor',c(icond,:),'LineWidth',0.75);
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',0:5);
    set(gca,'YTick',0:1:2,'YTickLabel',{'0','1.0','2.0'});
    xlabel('stimulus lag (trials)','FontSize',8);
    ylabel('stimulus weight','FontSize',8);
end

% print to file
if ~exist('./figs','dir')
    mkdir('./figs');
end
figure(hf);
axes = findobj(gcf,'type','axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
set(gcf,'PaperPositionMode','manual', ...
    'PaperPosition',[0.5*(21.0 -figw),0.5*(29.7-figh),figw,figh],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
if is_savetofile
    print(sprintf('./figs/fig_breg_%s_%s_%s',modname,sfit,teststr{itest}),'-painters','-dpdf');
end

%% Principal component analyses on behavioral measures (concatenated conditions)

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session        = 'test'; % set 'session' to either 'test' or 'retest'
is_sim_overlay = false;   % set to true to show simulated behavior
nexpl          = 4;      % number of PCs to analyze

is_savetofile  = false;   % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
isok = idx_incl_trt(:,itest);
nok  = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);

fprintf('Running PCA on concatenated behavioral measures over both conditions (%s)...\n',teststr{itest});

dtlist  = {'revt','swit','psy'};
c       = [.25 .56 .58; .9 .47 .18]; 
ylims   = [-.2 .2; 0 .3; -.21 0];
mrksz   = 10;

xsrc_cat = [];
zsrc_cat = [];
xsim_cat = [];
zsim_cat = [];
for dtype = dtlist
    % subjects
    xsrc     = eval(sprintf('p%s_sub_all',dtype{1}));
    xsrc_cnd = [xsrc(isok,:,1,itest) xsrc(isok,:,2,itest)];
    xsrc_cat = cat(2,xsrc_cat,xsrc_cnd);
    zsrc_cat = cat(2,zsrc_cat,zscore(xsrc_cnd,[],'all'));

    % model
    xsim     = eval(sprintf('p%s_sim_all',dtype{1}));
    xsim_cnd = [xsim(isok,:,1,itest) xsim(isok,:,2,itest)];
    xsim_cat = cat(2,xsim_cat,xsim_cnd);
    zsim_cat = cat(2,zsim_cat,zscore(xsim_cnd,[],'all'));
end

[coeffs,scores,~,~,expl] = pca(zsrc_cat);
[cf_sim,sc_sim,~,~,esim] = pca(zsim_cat);


% plot
hf(1) = figure('Color','white'); % figure with PC ingredients 
clf
hf(2) = figure('Color','white'); % figure with median split on PCs
clf
for idtype = 1:3
    dtype    = dtlist{idtype};
    xsrc     = eval(sprintf('p%s_sub_all',dtype));
    xsrc     = xsrc(isok,:,:);
    xsrc_opt = eval(sprintf('p%s_opt_all',dtype));
    xsrc_opt = xsrc_opt(isok,:,:);
    
    switch dtype
        case 'psy'
            xval        = linspace(-efac,+efac,11);
            xval        = 0.5*(xval(1:end-1)+xval(2:end));
            xticks      = -1.5:1:1.5;
            xticklabels = xticks;
            xlabelstr   = 'stimulus direction';
            xlims       = [min(xticks)-.5 max(xticks)+.5];
        otherwise
            xval        = 1:10;
            xticks      = [2 4 7 9];
            xticklabels = [-4 -2 2 4];
            xlabelstr   = 'position from reversal';
            xlims       = [min(xticks)-1.5 max(xticks)+1.5];
    end
    
    for ipc = 1:nexpl
        set(0,'CurrentFigure',hf(1));
        % ingredient coefficients in PCi
        ax1 = subplot(nexpl,3,3*(ipc-1)+(idtype), 'Parent', hf(1));
        hold(ax1, 'on');
        yline(0,'Color',0.8*[1,1,1],'LineWidth',0.75);
        xline((min(xval)+max(xval))/2,'Color','k')
        
        for icond = 1:2
            xdsp = (icond*.2-.3)/4 ; % x displacement for condition display (too much overlap)
            xrange = (1:10) + (icond-1)*10 + (idtype-1)*20;
            % model
            if is_sim_overlay
                plot(xval+xdsp,cf_sim(xrange,ipc),'-','LineWidth',1.50,'Color',c(icond,:));
            end
            % subjects
            if ~is_sim_overlay
                plot(xval+xdsp,coeffs(xrange,ipc),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
            end
            plot(xval+xdsp,coeffs(xrange,ipc),'wo', 'MarkerSize',4,'MarkerFaceColor',c(icond,:), 'LineWidth',0.75); % subject markers
        end
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[3,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        
        title(sprintf('%.2f%% expl.',expl(ipc)));
        ylabel('PC coefficient','FontSize',8);
        xlabel(xlabelstr,'FontSize',8);
        set(gca,'XTick',xticks,'XTickLabels',xticklabels);
        xlim(xlims);
        if ipc == 1
            yticks(-.2:.2:.4);
            ylim(ylims(idtype,:));
        end

        % 2nd figure
        set(0,'CurrentFigure',hf(2));
        ax2 = subplot(nexpl,3,3*(ipc-1)+(idtype), 'Parent', hf(2));
        hold(ax2, 'on');
        % median split view of the measure (on PCi)
        idx_upper = scores(:,ipc) >= median(scores(:,ipc));
        for icond = 1:2
            for imed = 1:2
                xdsp = (icond*.2-.3)/(4/imed) ; % x displacement for condition display (too much overlap)
                xrange = (1:10) + (icond-1)*10 + (idtype-1)*20;
                if imed == 1
                    idx_plot = ~idx_upper;
                else
                    idx_plot = idx_upper;
                end
                hold on
                yline(.5,'Color',[.8 .8 .8 .7],'LineWidth',.5);
                plot(xval,mean(xsrc_opt(:,:,icond)),'-','LineWidth',1.50,'Color',[1,1,1]*0.8); % optimal
                xavg = mean(xsrc(idx_plot,:,icond));
                xsem = std(xsrc(idx_plot,:,icond))/sqrt(sum(idx_plot));
                err = errorbar(xval+xdsp,xavg,xsem,'Color',c(icond,:),'LineStyle','none','CapSize',0,'LineWidth',.5);
                err.Line.ColorData(4) = 1-((imed-1)/2);
                plot(xval+xdsp,xavg,'o-','Color',c(icond,:), ...
                    'MarkerSize',4/imed,'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','w','LineWidth',0.75);
                xline(5.5,'Color',[0 0 0 .7],'LineWidth',.5);
            end    
        end
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[5/3,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        
        measstr = {'reversed','switched','repeated'};
        switch dtype
            case 'revt'
                imeas = 1;
                ylim([.1 .9]);
                yticks(.2:.2:.8);
            case 'swit'
                imeas = 2;
                ylim([.05 .5]);
                yticks(.1:.1:.4);
            case 'psy'
                imeas = 3;
                ylim([0 1]);
                yticks(0:.2:1);
        end
        ylabel(sprintf('fraction %s',measstr{imeas}),'FontSize',8);
        xlabel(xlabelstr,'FontSize',8);
        set(gca,'XTick',xticks,'XTickLabels',xticklabels);
        xlim(xlims);
    
        for ifig = 1:numel(hf)
            set(0,'CurrentFigure',hf(ifig));
            axes = findobj(gcf, 'type', 'axes');
            for a = 1:length(axes)
                if axes(a).YColor <= [1 1 1]
                    axes(a).YColor = [0 0 0];
                end
                if axes(a).XColor <= [1 1 1]
                    axes(a).XColor = [0 0 0];
                end
            end
            hold off
        end
    end
end

% save figures to file
if is_savetofile
    set(0,'CurrentFigure',hf(1));
    set(hf(1),'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,19,18],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    fname = sprintf('./figs/pca_ingredients_concatenated_conditions_%s',teststr{itest});
    print(fname,'-painters','-dpdf');
    
    set(0,'CurrentFigure',hf(2));
    set(hf(2),'PaperPositionMode','manual', ...
    'PaperPosition',[2.5,9,19,20],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
    figure(hf(2));
    fname = sprintf('./figs/fullpc%d_medsplit_concatenated_conditions_%s',ipc,teststr{itest});
    print(fname,'-painters','-dpdf');
end    
    
%% Principal component analyses on behavioral measures (concatenated sessions)
% 
% The included subjects for this analysis are by requirement, those of
% retest only

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
condition      = 'bandit'; % set 'condition' to either 'bandit' or 'fairy'
is_sim_overlay = false;     % set to true to show simulated behavior
is_savetofile  = true;     % set to true to save figure to file (in /figs/)

nexpl          = 4;        % number of PCs to analyze
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

isok = idx_incl_trt(:,2);
nok  = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);

if strcmpi(condition,'bandit')
    icond = 1;
elseif strcmpi(condition,'fairy')
    icond = 2;
end
nok  = sum(isok);
fprintf('Running PCA on concatenated behavioral measures over both sessions (%s)...\n',condstr{icond});
fprintf('Number of subjects in analysis : %d\n',nok);

dtlist  = {'revt','swit','psy'};
c       = [.25 .56 .58; .9 .47 .18];
ylims   = [-.2 .2; 0 .25; -.2 0];
mrksz   = 10;

xsrc_cat = [];
zsrc_cat = [];
xsim_cat = [];
zsim_cat = [];
for dtype = dtlist
    % subjects
    xsrc     = eval(sprintf('p%s_sub_all',dtype{1}));
    xsrc_cnd = [xsrc(isok,:,icond,1) xsrc(isok,:,icond,2)];
    xsrc_cat = cat(2,xsrc_cat,xsrc_cnd);
    zsrc_cat = cat(2,zsrc_cat,zscore(xsrc_cnd,[],'all'));

    % model
    xsim     = eval(sprintf('p%s_sim_all',dtype{1}));
    xsim_cnd = [xsim(isok,:,icond,1) xsim(isok,:,icond,2)];
    xsim_cat = cat(2,xsim_cat,xsim_cnd);
    zsim_cat = cat(2,zsim_cat,zscore(xsim_cnd,[],'all'));
end

[coeffs,scores,~,~,expl] = pca(zsrc_cat);
[cf_sim,sc_sim,~,~,esim] = pca(zsim_cat);

% plot
hf(1) = figure('Color','white'); % figure with PC ingredients 
clf
hf(2) = figure('Color','white'); % figure with median split on PCs
clf
for idtype = 1:3
    dtype    = dtlist{idtype};
    xsrc     = eval(sprintf('p%s_sub_all',dtype));
    xsrc     = xsrc(isok,:,:);
    xsrc_opt = eval(sprintf('p%s_opt_all',dtype));
    xsrc_opt = xsrc_opt(isok,:,:);
    
    switch dtype
        case 'psy'
            xval        = linspace(-efac,+efac,11);
            xval        = 0.5*(xval(1:end-1)+xval(2:end));
            xticks      = -1.5:1:1.5;
            xticklabels = xticks;
            xlabelstr   = 'stimulus direction';
            xlims       = [min(xticks)-.5 max(xticks)+.5];
        otherwise
            xval        = 1:10;
            xticks      = [2 4 7 9];
            xticklabels = [-4 -2 2 4];
            xlabelstr   = 'position from reversal';
            xlims       = [min(xticks)-1.5 max(xticks)+1.5];
    end
    
    for ipc = 1:nexpl
        set(0,'CurrentFigure',hf(1));
        % ingredient coefficients in PCi
        ax1 = subplot(nexpl,3,3*(ipc-1)+(idtype), 'Parent', hf(1));
        hold(ax1,'on');
        yline(0,'Color',0.8*[1,1,1],'LineWidth',0.75);
        xline((min(xval)+max(xval))/2,'Color','k')
        
        for itest = 1:2
            mrksp = 'o-';
            if itest == 2
                mrksp = 'd-';
            end
            xdsp = (itest*.2-.3)/4 ; % x displacement for condition display (too    much overlap)
            xrange = (1:10) + (itest-1)*10 + (idtype-1)*20;
            % model
            if is_sim_overlay
                plot(xval+xdsp,cf_sim(xrange,ipc),'-','LineWidth',1.50,'Color',c(icond,:));
            end
            % subjects
            if ~is_sim_overlay
                plot(xval+xdsp,coeffs(xrange,ipc),'Color',c(icond,:),'LineWidth',0.75); % connecting lines
            end
            plot(xval+xdsp,coeffs(xrange,ipc),mrksp(1),'MarkerSize',4,'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','w','LineWidth',0.75); % subject markers
        end
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[3,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        
        title(sprintf('%.2f%% expl.',expl(ipc)));
        ylabel('PC coefficient','FontSize',8);
        xlabel(xlabelstr,'FontSize',8);
        set(gca,'XTick',xticks,'XTickLabels',xticklabels);
        xlim(xlims);
        if ipc == 1
            yticks(-.2:.2:.4);
            ylim(ylims(idtype,:));
        end

        % 2nd figure
        set(0,'CurrentFigure',hf(2));
        ax2 = subplot(nexpl,3,3*(ipc-1)+(idtype), 'Parent', hf(2));
        hold(ax2, 'on');
        % median split view of the measure (on PCi)
        idx_upper = scores(:,ipc) >= median(scores(:,ipc));
        for itest = 1:2
            mrksp = 'o-';
            if itest == 2
                mrksp = 'd-';
            end
            for imed = 1:2
                xdsp = (itest*.2-.3)/(4/imed) ; % x displacement for condition display (too much overlap)
                xrange = (1:10) + (itest-1)*10 + (idtype-1)*20;
                if imed == 1
                    idx_plot = ~idx_upper;
                else
                    idx_plot = idx_upper;
                end
                hold on
                yline(.5,'Color',[.8 .8 .8 .7],'LineWidth',.5);
                plot(xval,mean(xsrc_opt(:,:,itest)),'-','LineWidth',1.50,'Color',[1,1,1]*0.8); % optimal
                xavg = mean(xsrc(idx_plot,:,itest));
                xsem = std(xsrc(idx_plot,:,itest))/sqrt(sum(idx_plot));
                err  = errorbar(xval+xdsp,xavg,xsem,'Color',c(icond,:),'LineStyle','none','CapSize',0,'LineWidth',.5);
                err.Line.ColorData(4) = 1-((imed-1)/2);
                plot(xval+xdsp,xavg,mrksp,'Color',c(icond,:), ...
                    'MarkerSize',4/imed,'MarkerFaceColor',c(icond,:),'MarkerEdgeColor','w','LineWidth',0.75);
                xline(5.5,'Color',[0 0 0 .7],'LineWidth',.5);
            end    
        end
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[5/3,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        
        measstr = {'reversed','switched','repeated'};
        switch dtype
            case 'revt'
                imeas = 1;
                ylim([.1 .9]);
                yticks(.2:.2:.8);
            case 'swit'
                imeas = 2;
                ylim([.05 .5]);
                yticks(.1:.1:.4);
            case 'psy'
                imeas = 3;
                ylim([0 1]);
                yticks(0:.2:1);
        end
        ylabel(sprintf('fraction %s',measstr{imeas}),'FontSize',8);
        xlabel(xlabelstr,'FontSize',8);
        set(gca,'XTick',xticks,'XTickLabels',xticklabels);
        xlim(xlims);
    
        for ifig = 1:2
            set(0,'CurrentFigure',hf(ifig));
            axes = findobj(gcf, 'type', 'axes');
            for a = 1:length(axes)
                if axes(a).YColor <= [1 1 1]
                    axes(a).YColor = [0 0 0];
                end
                if axes(a).XColor <= [1 1 1]
                    axes(a).XColor = [0 0 0];
                end
            end
            hold off
        end
    end
end

% save figures to file
if is_savetofile
    set(0,'CurrentFigure',hf(1));
    set(hf(1),'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,19,18],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    fname = sprintf('./figs/pca_ingredients_concatenated_sessions_%s',condstr{icond});
    print(fname,'-painters','-dpdf');
    
    set(0,'CurrentFigure',hf(2));
    set(hf(2),'PaperPositionMode','manual', ...
    'PaperPosition',[2.5,9,19,20],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
    fname = sprintf('./figs/fullpc%d_medsplit_concatenated_sessions_%s',ipc,condstr{icond});
    print(fname,'-painters','-dpdf');
end    

%% Model parameter fits
% Recall: 1/ The variable 'pars' holds the model parameter fits for all fitted
%             subjects in both conditions and sessions.
%         2/ The identity of each parameter (i.e., dimension 2) is described
%             in the variable 'parstr'.
%         3/ Log-transformed parameter values are in 'pars_logt'
%         Dimensions: (subj) x (parameter) x (condition) x (session)

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
is_savetofile = false;     % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~
parorder_plot = [3 4 1 2 5 6];
datwid = 0.2;
mrksz  = 10;
c = [.25 .56 .58; .9 .47 .18]; 
parrgb = [109 195 195; 47 123 162; 100 179 100; 
          166 88 67; 109 207 244; 186 126 142; 186 126 180]/255;
parstr_full = {'hazard rate','prior compression','stimulus compression',...
               'stimulus bias','inference noise','evidence bias'};
lims = [0 .6; -1 3; 0 5.5; -.4 .2; 0 4; -1 .5];
linc = [.25 1 2 .2 1 .5];

clearvars hf1 hf2 hf_sess ax1 ax2 ax
close all
hf1(1) = figure('Color','white'); clf % violin plot of parameters in both conditions (test)
hf1(2) = figure('Color','white'); clf % violin plot of parameters in both conditions (retest)
hf2(1) = figure('Color','white'); clf % scatter plot of parameters between conditions (test)
hf2(2) = figure('Color','white'); clf % scatter plot of parameters between conditions (retest)
hf_sess(1) = figure('Color','white'); clf % scatter plot of parameters between sessions (bandit)
hf_sess(2) = figure('Color','white'); clf % scatter plot of parameters between sessions (fairy)
for ipar = 1:npar
    fprintf('Using Wilcoxon signed rank tests to test differences from optimal...\n');
    for itest = 1:2
        isok = idx_incl_trt(:,itest);
        pars_plt = pars(isok,:,:,:);
        
        % significance from zero test
        for icond = 1:ncnd
            if ipar == 1
                [p,~,stats] = signrank(pars_plt(:,ipar,icond,itest),h_opt);
                fprintf('%s(%s): p=%.4f, Z=%.4f\n',parstr{ipar},pad(condstr{icond},6),p,stats.zval);
            else
                [p,~,stats] = signrank(pars_plt(:,ipar,icond,itest));
                fprintf('%s(%s): p=%.4f, Z=%.4f\n',parstr{ipar},pad(condstr{icond},6),p,stats.zval);
            end
        end
        
        % condition-wise test
        fprintf('Between-condition difference in %s: ',teststr{itest});
        [p,~,stats] = signrank(pars_plt(:,ipar,2,itest),pars_plt(:,ipar,1,itest));
        fprintf('Z=%.3f, p=%.3f (%s)\n',stats.zval,p,parstr{ipar});
    
        % figure 1: violin plot of parameters in both conditions
        set(0,'CurrentFigure',hf1(itest));
        ax1(itest) = subplot(3,3,parorder_plot(ipar),'Parent',hf1(itest));
        hold(ax1(itest), 'on');
        xticks(1:2);
        xloc = nan(size(pars_plt,1),2);
        yloc = nan(size(pars_plt,1),2);
        for icond = 1:ncnd
            % plan scattered violin 
            x    = pars_plt(:,ipar,icond,itest);
            xk   = linspace(min(x),max(x),100);
            pk   = ksdensity(x,xk);
            pk   = pk/max(pk);
            str  = interp1(xk,pk,x);
            jit  = linspace(-1,+1,numel(x));
            xpos = nan(size(x));
            for j = randperm(numel(x))
                xpos(j) = icond+jit(j)*str(j)*datwid;
            end
            % connecting lines
            xloc(:,icond) = xpos;
            yloc(:,icond) = x;
            if icond == ncnd
                plot(xloc',yloc','-','LineWidth',0.25,'Color',[1 1 1 .5]*.8);
            end
            % scattered violin
            p = scatter(xpos,x,mrksz*.5, ...
                'MarkerFaceColor',parrgb(ipar,:),'MarkerEdgeColor','none');
            p.MarkerFaceAlpha = 0.2;
            plot([1 1]*icond,std(x)/sqrt(sum(isok)),'Color',c(icond,:),'LineWidth',1); % error bars (SEM)
            scatter(icond,mean(x),mrksz*2,[1 1 1],'LineWidth',1,'MarkerFaceColor',c(icond,:)); % means
        end
        pbar = 1/2*(3+0.2)/2.2; % plot box aspect ratio
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        xlabel('condition','FontSize',8);
        ylabel(sprintf('%s',parstr_full{ipar}),'FontSize',8);
        clearvars xticklabels xticks
        xticklabels(condstr(1:ncnd));
        xlim([.5 ncnd+.5]);
        ylim(lims(ipar,:));
        hold(ax1(itest), 'off');
    
        % figure 2: scatter plot of parameters in both conditions
        set(0,'CurrentFigure',hf2(itest));
        ax2(itest) = subplot(3,3,parorder_plot(ipar),'Parent',hf2(itest));
        hold(ax2(itest), 'on');
        x = pars_plt(:,ipar,1,itest);
        y = pars_plt(:,ipar,2,itest);
        xlim(lims(ipar,:));
        ylim(lims(ipar,:));
        p = scatter(x,y,mrksz*.5, ...
                'MarkerFaceColor',parrgb(ipar,:),'MarkerEdgeColor','none');
        p.MarkerFaceAlpha = 0.2;
        [r,p] = corr(x,y,'Type','Pearson');
        fprintf('Between-condition correlation in %s: ',teststr{itest});
        fprintf('%s: r=%.4f, p=%.4f\n',parstr{ipar},r,p);
        xrange = min(x):.01:max(x);
        [pn,s] = polyfit(x,y,1);
        [py,d] = polyconf(pn,xrange,s,'alpha',0.05,'predopt','curve');
        s = shadedErrorBar(xrange,py,d,'patchSaturation',.1,'lineprops',{'LineWidth',1,'Color',parrgb(ipar,:)});
        set(s.edge,'LineStyle','none');
        s.HandleVisibility = 'off';
        xticks(lims(ipar,1):linc(ipar):lims(ipar,2));
        yticks(lims(ipar,1):linc(ipar):lims(ipar,2));
        title(parstr{ipar});
    
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1,1,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02/.7273);
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        xlabel(condstr{1},'FontSize',8);
        ylabel(condstr{2},'FontSize',8);
        hold(ax2(itest), 'off');

        % figure 3: between-session correlation 
        if itest == 2
            for icond = 1:ncnd
                set(0,'CurrentFigure',hf_sess(icond));
                ax3(icond) = subplot(3,3,parorder_plot(ipar),'Parent',hf_sess(icond));
                hold(ax3(icond), 'on');
                x = pars_plt(:,ipar,icond,1);
                y = pars_plt(:,ipar,icond,2);

                xlim(lims(ipar,:));
                ylim(lims(ipar,:));
                p = scatter(x,y,mrksz*.5, ...
                        'MarkerFaceColor',parrgb(ipar,:),'MarkerEdgeColor','none');
                p.MarkerFaceAlpha = 0.2;
                [r,p] = corr(x,y,'Type','Pearson');
                fprintf('Between-session correlation in %s: ',condstr{icond});
                fprintf('%s: r=%.4f, p=%.4f\n',parstr{ipar},r,p);
                xrange = min(x):.01:max(x);
                [pn,s] = polyfit(x,y,1);
                [py,d] = polyconf(pn,xrange,s,'alpha',0.05,'predopt','curve');
                s = shadedErrorBar(xrange,py,d,'patchSaturation',.1,'lineprops',{'LineWidth',1,'Color',c(icond,:)});
                set(s.edge,'LineStyle','none');
                s.HandleVisibility = 'off';
                xticks(lims(ipar,1):linc(ipar):lims(ipar,2));
                yticks(lims(ipar,1):linc(ipar):lims(ipar,2));
                title(parstr{ipar});
            
                set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1,1,1]);
                set(gca,'TickDir','out','TickLength',[1,1]*0.02/.7273);
                set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
                xlabel(teststr{1},'FontSize',8);
                ylabel(teststr{2},'FontSize',8);
                hold(ax3(icond), 'off');
            end
        end
    end 
end

if is_savetofile
    % save figure to file
    for jfig = 1:2
        set(0,'CurrentFigure',hf1(jfig));
        pause(.5);
        set(hf1(jfig),'PaperPositionMode','manual', ...
            'PaperPosition',[2.5,13,16,14],'PaperUnits','centimeters', ...
            'PaperType','A4','PaperOrientation','portrait');
        fname = sprintf('./figs/parameters_%s',teststr{jfig});
        print(fname,'-painters','-dpdf');
    
        % save figure to file
        set(0,'CurrentFigure',hf2(jfig));
        pause(.5);
        set(hf2(jfig),'PaperPositionMode','manual', ...
            'PaperPosition',[2.5,11,13,11],'PaperUnits','centimeters', ...
            'PaperType','A4','PaperOrientation','portrait');
        fname = sprintf('./figs/parameters_correlation_between_condition_%s',teststr{jfig});
        print(fname,'-painters','-dpdf');
    
        % save figure to file
        set(0,'CurrentFigure',hf_sess(jfig));
        pause(.5);
        set(hf_sess(jfig),'PaperPositionMode','manual', ...
            'PaperPosition',[2.5,11,13,11],'PaperUnits','centimeters', ...
            'PaperType','A4','PaperOrientation','portrait');
        fname = sprintf('./figs/parameters_correlation_between_session_%s',condstr{jfig});
        print(fname,'-painters','-dpdf');
    end
end    

%% Parameter correlations in test-retest vs bandit-fairy

corrfn = @(x1,x2) corr(x1,x2,'Type','Pearson', 'tail', 'both');
nsamp = 1e4;

hf = figure('Color','white'); 
for ipar = 1:npar
    subplot(1,npar,ipar);
    % test-retest
    isok = idx_incl_trt(:,2);
    % in bandit
    [r_trt_b,p] = corr(pars(isok,ipar,1,2), pars(isok,ipar,1,1),'Type','Pearson');
    ci_trt_b = bootci(nsamp,{corrfn,pars(isok,ipar,1,2),pars(isok,ipar,1,1)});
    fprintf('r=%.4f, p=%.4f (%s)\n',r_trt_b,p,parstr{ipar});
    % in fairy
    [r_trt_f,p] = corr(pars(isok,ipar,2,2), pars(isok,ipar,2,1),'Type','Pearson');
    ci_trt_f = bootci(nsamp,{corrfn,pars(isok,ipar,2,2),pars(isok,ipar,2,1)});
    fprintf('r=%.4f, p=%.4f (%s)\n',r_trt_f,p,parstr{ipar});

    % condition-wise
    % in test
    isok = idx_incl_trt(:,1);
    [r_bf_t,p] = corr(pars(isok,ipar,2,1), pars(isok,ipar,1,1),'Type','Pearson');
    ci_bf_t = bootci(nsamp,{corrfn,pars(isok,ipar,2,1),pars(isok,ipar,1,1)});
    fprintf('r=%.4f, p=%.4f (%s)\n',r_bf_t,p,parstr{ipar});
    % in retest
    isok = idx_incl_trt(:,2);
    [r_bf_r,p] = corr(pars(isok,ipar,2,2), pars(isok,ipar,1,2),'Type','Pearson');
    ci_bf_r = bootci(nsamp,{corrfn,pars(isok,ipar,2,2),pars(isok,ipar,1,2)});
    fprintf('r=%.4f, p=%.4f (%s)\n',r_bf_r,p,parstr{ipar});
    
    rs = [r_bf_t,r_bf_r];

    b = barh(flip(rs),'FaceColor',parrgb(ipar,:),'EdgeColor','none');
    b.FaceAlpha = 0.7;
    hold on
    plot(ci_bf_t,[2 2],'Color',parrgb(ipar,:));
    plot(ci_bf_r,[1 1],'Color',parrgb(ipar,:));
    xline(r_trt_b,':','Color',condrgb(1,:),'LineWidth',1.5);
    plot(ci_trt_b,[2.5 2.5],'Color',condrgb(1,:));
    xline(r_trt_f,':','Color',condrgb(2,:),'LineWidth',1.5);
    plot(ci_trt_f,[2.3 2.3],'Color',condrgb(2,:));
    xline(0);
    xticks(-.4:.2:.8);
    xtickangle(0);
    xlim([-.4 .8]);
    ylim([0.25 2.75]);
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1.5,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/.7273);
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
end

set(hf,'PaperPositionMode','manual', ...
    'PaperPosition',[0,11,20,13],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
fname = sprintf('./figs/par_correlations_all');
print(fname,'-painters','-dpdf');

%% PCA and parameters

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
ana_type       = 'trt';   % 'cond' or 'trt'
session        = 'test';   % set 'session' to either 'test' or 'retest'
condition      = 'fairy'; % or 'fairy
is_sim_overlay = false;    % set to true to show simulated behavior
nexpl          = 4;        % number of PCs to analyze

is_savetofile  = false;   % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

if strcmpi(ana_type,'cond')
    isok = idx_incl_trt(:,itest);
elseif strcmpi(ana_type,'trt')
    isok = idx_incl_trt(:,2);
else
    error('Invalid analysis type!')
end

if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
if strcmpi(condition,'bandit')
    icond = 1;
elseif strcmpi(condition,'fairy')
    icond = 2;
end 


nok  = sum(isok);
fprintf('Number of subjects in analysis : %d\n',nok);
fprintf('Running PCA on concatenated behavioral measures over both conditions (%s)...\n',teststr{itest});

dtlist  = {'revt','swit','psy'};
xsrc_cat = [];
zsrc_cat = [];
for dtype = dtlist
    % subjects
    xsrc = eval(sprintf('p%s_sub_all',dtype{1}));
    if strcmpi(ana_type,'cond')
        xsrc_cnd = [xsrc(isok,:,1,itest) xsrc(isok,:,2,itest)];
    elseif strcmpi(ana_type,'trt')
        xsrc_cnd = [xsrc(isok,:,icond,1) xsrc(isok,:,icond,2)];
    end
    xsrc_cat = cat(2,xsrc_cat,xsrc_cnd);
    zsrc_cat = cat(2,zsrc_cat,zscore(xsrc_cnd,[],'all'));
end


nok  = sum(isok);
fprintf('Running PCA on concatenated behavioral measures over both sessions (%s)...\n',condstr{icond});
fprintf('Number of subjects in analysis : %d\n',nok);


[~,scores,~,~,expl] = pca(zsrc_cat);
pars_filt = pars(isok,:,icond,itest);

clf
for ipar = 1:npar
    subplot(3,3,ipar)
    hold on
    x = scores(:,1);
    y = pars_filt(:,ipar);
    scatter(x,y);
    [r,p] = corr(x,y,'Type','Spearman');
    fprintf('%s: r=%.3f, p=%.3f\n',parstr{ipar},r,p);
end

%% Correlation matrix of parameters
%% Note on the following analyses involving ICAR 
%  The ICAR scores for all subjects are contained in the variable 'icar'
%   where the 1st column refers to test, and the 2nd, to retest. 
%  
%  RECALL:
%  'idx_icar_concat' : Indices of subjects who have fully given ICAR 
%                       responses and passed the attention check
%  'idx_ques_fail'   : Indices of subjects who failed the attention check
%  'idx_subj_concat' : Indices of subjects with full behavioral data
%  'idx_incl_trt'    : Indices of subjects who were best fitted with the
%                       hidden-state inference model in test and retest

%% Overall distribution of ICAR scores in test and stability in retest

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
session        = 'test'; % set 'session' to either 'test' or 'retest'
is_savetofile  = true;   % set to true to save figure to file (in /figs/)
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~
if strcmpi(session,'test')
    itest = 1;
elseif strcmpi(session,'retest')
    itest = 2;
end
isok    = idx_icar_concat(:,1) & idx_subj_concat(:,1); 
isok_rt = isok & idx_icar_concat(:,2) & idx_subj_concat(:,2);
nok     = sum(isok);
nok_rt  = sum(isok_rt);
nfail   = sum(idx_ques_fail(:,1));
x       = icar(isok,1);

fprintf('Number of participants who completed the ICAR: %d\n', nok+nfail);
fprintf('Number of participants who successfully completed ICAR: %d\n', nok);
fprintf('Number of participants who failed attention checks: %d\n', nfail);
fprintf('Number of participants from inclusion in retest: %d\n', nok_rt);

mrksz  = 10;
figure('Color','white'); clf 
% histogram of ICAR scores in the test session
subplot(1,2,1); 
hold on
n = hist(x,0:16);
fprintf('Mean ICAR: %.2f ± %.2f\n',mean(x),std(x));

t_stat = (mean(x)-8.00)/sqrt((std(x)^2/numel(x)) + (3.64^2/434)); % from Merz, Lace & Eisenstein 2022
df = numel(x) + 434 - 2;
p = 2*(1-tcdf(abs(t_stat),df));
fprintf('2-tailed t-test sample mean vs larger sample mean: t=%.3f, p=%.4f\n',t_stat,p);

bar(0:16,n,1,'EdgeColor','w','LineWidth',0.75,'FaceColor',[0.8,0.8,0.8]);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1.2,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02);
set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
set(gca,'XTick',0:4:16);
set(gca,'YTick',0:10:30);
xlabel('ICAR-16 score','FontSize',8);
ylabel('number of participants','FontSize',8);
xlim([-0.5,16.5]);
ylim([0,30]);

% scatter plot of ICAR scores in test and retest (i.e., reliability)
subplot(1,2,2);
x = icar(isok_rt,1);
y = icar(isok_rt,2);
subplot(1,2,2);
hold on
plot([0 16],[0 16],'Color','k','LineStyle',':'); % identity line
xeps = normrnd(0,.1,size(x)); % jitter
yeps = normrnd(0,.1,size(x)); % jitter
scatter(x+xeps,y+yeps,mrksz*.5,'MarkerFaceColor',[61 80 81]/100, ...
        'MarkerFaceAlpha',.8,'MarkerEdgeColor','none'); % ICAR scores
% linear fit
xrange = min(x):.01:max(x);
[pn,s] = polyfit(x,y,1);
[py,d] = polyconf(pn,xrange,s,'alpha',0.05,'predopt','curve');
s = shadedErrorBar(xrange,py,d,'patchSaturation',.1,'lineprops',{'LineWidth',1,'Color',[61 80 81]/100});
set(s.edge,'LineStyle','none');
s.HandleVisibility = 'off';
hold off
[r,p] = corr(x,y); % Pearson correlation
fprintf('Correlation ICAR scores between test and retest: r=%+.3f, p=%.4f\n',r,p);
[icc,lb,ub,~,~,~,p] = ICC(icar(isok_rt,:),'A-1'); % ICC A-1
fprintf('ICC A-1: r=%.3f [%.3f %.3f], p=%.4f\n',icc,lb,ub,p);
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1.2,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(1,1));
set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
set(gca,'XTick',0:4:16);
set(gca,'YTick',0:4:16);
xlabel('ICAR-16 (test)','FontSize',8);
ylabel('ICAR-16 (retest)','FontSize',8);
xlim([-0.5,16.5]);
ylim([-0.5,16.5]);

if is_savetofile  
    axes = findobj(gcf,'type','axes');
    for a = 1:length(axes)
        if axes(a).YColor <= [1 1 1]
            axes(a).YColor = [0 0 0];
        end
        if axes(a).XColor <= [1 1 1]
            axes(a).XColor = [0 0 0];
        end
    end
    set(gcf,'PaperPositionMode','manual', ...
        'PaperPosition',[0.5*(21.0 -12),0.5*(29.7-13),10,10],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    print(sprintf('./figs/icar_basics'),'-painters','-dpdf');
end

%% Run ICAR-16 dependent multinomial regression model (REQ'D)
%
%  This cell computes predictions from the ICAR-16 dependent multinomial
%  regression model, its associated p-value by comparing it to the baseline
%  regression model, and empirical fractions of each label wrt ICAR-16 bins.
%
%  This works for one condition at a time, set using icond. You can choose to
%  include both sessions (test and retest) at once, or only one of them. When
%  both sessions are included, this code pools subjects across sessions without
%  considering that some of the subjects have been tested and retested.

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
isess = 1; %:2; % session/sessions of interest (1=test 2=retest, 1:2=both)
nres  = 1e3; % number of bootstrap resamples
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

rgb = [82 73 85; 97 81 63; 61 80 81]/100; 
if ~exist('icaq','var')
    fprintf('Processing ICAR data for analysis...\n');
    ques = cell(1,2);
    mini = cell(1,2);
    icaq = nan(nsubj,2,16); % ICAR-16 responses to individual questions
    for isess = 1:2
        if isess == 1
            samp_sess = samp;
        else
            samp_sess = sprintf('%s_retest',samp);
        end
        fprintf('Processing data in %s...\n',teststr{isess});
        load(sprintf('./processed/%s/ques_struct_%s.mat',samp_sess,samp_sess),'ques_struct');
        ques{isess} = ques_struct;
        for isubj = 1:nsubj
            if ~isempty(ques{isess}{isubj}) && isfield(ques{isess}{isubj}.iq,'score')
                icar(isubj,isess) = ques{isess}{isubj}.iq.score;
                icaq(isubj,isess,:) = ques{isess}{isubj}.iq.raw;
            end
        end
    end
end

hf1 = figure('Color','white'); clf 
for icond = 1:2
    switch icond
        case 1, fprintf('Processing bandit condition...\n');
        case 2, fprintf('Processing fairy condition...\n');
        otherwise, error('Undefined condition number!');
    end
    
    % set label (1=null 2=stim 3=hinf)
    c1 = reshape(is_null_trt(:,icond,isess),[],1);
    c2 = reshape(is_stim_trt(:,icond,isess),[],1);
    c3 = reshape(is_hinf_trt(:,icond,isess),[],1);
    c  = c1*1+c2*2+c3*3;
    
    % set ICAR-16 as predictor
    x = reshape(icar(:,isess),[],1); % score
    q = reshape(icaq(:,isess,:),[],16); % responses to individual questions
    
    % remove subjects with missing predictor or label
    i = ~isnan(x) & c > 0;
    x = x(i);
    c = c(i);
    n = numel(x);
    fprintf('%d subjects included in analysis.\n',n);
    
    % compute predictions from multinomial regression model
    fprintf('Computing predictions from multinomial regression model...\n');
    xhat = (0:16)';
    for ires = 1:nres
        isub = randsample(n,n,true);
        b = mnrfit(x(isub),c(isub),'model','nominal','interactions','on');
        phat(:,:,ires) = mnrval(b,xhat,'model','nominal','interactions','on');
    end
    pmed = median(phat,3); % median
    pc95 = quantile(phat,[0.025,0.975],3); % 95% CI
    
    % compute p-value for the ICAR-16 dependent model
    % check https://en.wikipedia.org/wiki/Deviance_(statistics) for more information
    % about the deviance metric
    [~,dev1,stat1] = mnrfit(x,c,'model','nominal','interactions','on');
    [~,dev0,stat0] = mnrfit([],c,'model','nominal','interactions','on');
    pval = 1-chi2cdf(dev0-dev1,stat0.dfe-stat1.dfe);
    fprintf('p-value of ICAR-16 dependent multinomial regression model = %.4f\n',pval);
    
    % compute empirical fractions of each label wrt ICAR-16 bins
    fprintf('Computing empirical fractions of each label wrt ICAR-16 bins...\n');
    xmin = [0,4:12,13]; % inclusive lower bounds of each bin
    xmax = [3,4:12,16]; % inclusive upper bounds of each bin
    nbin = numel(xmin);
    xbin = [2,4:12,14]';
    pbin = nan(nbin,3); % empirical fraction of each label in each bin
    kbin = nan(nbin,1); % number of data points in each bin
    for ibin = 1:numel(xmin)
        for ic = 1:3
            pbin(ibin,ic) = mean(c(x >= xmin(ibin) & x <= xmax(ibin)) == ic);
            kbin(ibin) = nnz(x >= xmin(ibin) & x <= xmax(ibin));
        end
    end
    
    % plot predictions of ICAR-16 dependent multinomial regression model
    subplot(1,2,icond);
    hold on
    for ic = 1:3
        patch([xhat;flip(xhat)],[pc95(:,ic,1);flip(pc95(:,ic,2))],0.5*(rgb(ic,:)+1),'EdgeColor','none','FaceAlpha',0.5);
        plot(xhat,pmed(:,ic),'-','LineWidth',1.5,'Color',rgb(ic,:));
        plot(xbin,pbin(:,ic),'wo','MarkerSize',4,'MarkerFaceColor',rgb(ic,:),'LineWidth',0.75);
    end
    hold off
    set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[4/3,1,1]);
    set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(4/3,1));
    set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
    set(gca,'XTick',0:4:16);
    set(gca,'YTick',0:0.2:1);
    xlim([-1 17]);
    xlabel('ICAR-16 score','FontSize',8);
    ylabel('fraction of participants','FontSize',8);
end

axes = findobj(gcf,'type','axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
set(gcf,'PaperPositionMode','manual', ...
    'PaperPosition',[0.5*(21.0 -12),0.5*(29.7-13),12,13],'PaperUnits','centimeters', ...
    'PaperType','A4','PaperOrientation','portrait');
print(sprintf('./figs/fig_icar_mnrfit_sess%s',mat2str(isess)),'-painters','-dpdf');
fprintf('Done.\n');

% fraction of participants using some strategy in a condition
nsubj_ana = size(c,1);
nstrats = [sum(is_hinf_trt(:,:,1));  % number of subjects using hinf
           sum(is_stim_trt(:,:,1));  % number of subjects using stim
           sum(is_null_trt(:,:,1))]; % number of subjects using rand
cs = (nstrats/nsubj_ana)';
fprintf('Strategy: hinf | stim | rand\n');
fprintf('           %d     %d     %d  (bandit)\n',nstrats(1),nstrats(2),nstrats(3));
fprintf('           %d    %d     %d  (fairy)\n',  nstrats(4),nstrats(5),nstrats(6));

%% ICAR and model parameter fits
%  Computes normalized rank regression predicting ICAR scores from fitted 
%   parameters.

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
errortype = 'se'; % 'se':standard error, 'bs':95% bootstrapped CI
is_savetofile = false;
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

clc
close all
is_bserror = false;
switch errortype
    case 'se'
        fprintf('Error bars are standard errors.\n');
    case 'bs'
        fprintf('Error bars are bootstrapped 95%% confidence intervals.\n');
        is_bserror = true;
    otherwise
        error('Invalid value for "errortype"!');
end
parrgb = [109 195 195; 47 123 162; 100 179 100; 
          166 88 67; 109 207 244; 186 126 142; 186 126 180]/255;
age = demotable.age;

fprintf('Single regression of ICAR-16 score against all parameters\n');
hf = figure('Color','white'); clf 
ctr = 1;
pars_zscored = zscore(pars,[],2);
for isess = 1:2
    % Note: 'isok' can be changed to include a different subset of subjects
    %       By default it is set to all participants using hidden-state
    %       inference in the 1st session, and whether they had performed
    %       the ICAR in the session of interest
    isok = all(is_hinf_trt(:,:,1),2) & idx_icar_concat(:,isess);
    nok = sum(isok);

    fprintf('\nComputing regressions for subjects in %s...\n',teststr{isess});
    fprintf('%d subjects included in analysis.\n',nok);
    for icond = 1:2
        fprintf('For the %s condition...\n',condstr{icond});
        y    = icar(isok,isess); % get ICAR-16 score
        ages = age(isok);
        xpar = pars(isok,:,icond,isess); % get parameters
        xpar = [xpar ages];
        ndat = numel(y);
        
        % compute normalized ranks for each parameter
        xrnk = nan(ndat,npar);
        for ipar = 1:npar
            xrnk(:,ipar) = tiedrank(xpar(:,ipar));
        end
        xrnk = (xrnk-1)/(ndat-1); % normalize
        xdat = xrnk;

        % normalized rank regression
        [~,~,stat] = glmfit(xdat,y,'normal');
        beta       = stat.beta(2:end);
        tval       = stat.t(2:end);
        pval       = stat.p(2:end);
        serr       = stat.se(2:end);

        % individual correlations
        parstr_pad = cellfun(@(x)pad(x,6),parstr(parorder_plot),'UniformOutput',false);
        fprintf('%s         ',parstr_pad{:});
        fprintf('\n');
        arrayfun(@(x,y)fprintf('%.3f (%.4f) ',x,y),beta(parorder_plot), pval(parorder_plot));
        fprintf(' Regression betas and p-values \n');
        [rho,pval] = corr(xdat,y);
        arrayfun(@(x,y)fprintf('%.3f (%.3f) ',x,y),rho(parorder_plot), pval(parorder_plot));
        fprintf(' Pearson corr between normalized ranks and ICAR \n');
        [rho,pval] = corr(xpar,y,'Type','Spearman');
        arrayfun(@(x,y)fprintf('%.3f (%.3f) ',x,y),rho(parorder_plot), pval(parorder_plot));
        fprintf(' Spearman corr between raw parameter values and ICAR \n');

        % bootstrap error bars
        if is_bserror
            nsamp = 1e3;
            beta_bs = nan(nsamp,size(beta,1));
            if icond == 2
                fprintf('Bootstrapping error bars for beta regression weights...\n');
            end
            for i = 1:nsamp
                idx = randsample(ndat,ndat,true);
                [~,~,stat] = glmfit(xdat(idx,:),y(idx),'normal');
                beta_bs(i,:) = stat.beta(2:end)';
            end
            ci95 = quantile(beta_bs,[.025 .975])'; % 95% bootstrapped confidence interval   
            eneg = beta(parorder_plot)+ci95(parorder_plot,1);
            epos = beta(parorder_plot)+ci95(parorder_plot,2);
        end

        subplot(2,2,ctr);
        hold on
        ord = flip(parorder_plot);
        hb = barh(1:npar,flip(beta(parorder_plot)),'EdgeColor','w','FaceColor','flat','LineWidth',0.75);
        for ipar = 1:npar
            hb.CData(ipar,:) = parrgb(ord(ipar),:);
            hb.FaceAlpha = .8;
        end
        if strcmpi(errortype,'bs')
            er = errorbar(flip(beta(parorder_plot)),1:npar,eneg,epos,'horizontal','Marker','none', ...
                          'LineStyle','none','Color',[.3 .3 .3],'CapSize',0);
        else
            er = errorbar(flip(beta(parorder_plot)),1:npar,serr,'horizontal','Marker','none', ...
                          'LineStyle','none','Color',[.3 .3 .3],'CapSize',0);
            xlim([-6 2]);
        end
        xline(0,'Color',[.5 .5 .5],'LineWidth',1);
        xlabel('regression weight');
        set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[1,1.3,1]);
        set(gca,'TickDir','out','TickLength',[1,1]*0.02);
        set(gca,'FontName','Helvetica','FontSize',7.2,'LineWidth',0.75);
        set(get(gca, 'YAxis'), 'Visible', 'off');
        title(sprintf('%s %s',teststr{isess},condstr{icond}));
        hold off

        ctr = ctr + 1;
    end
end
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes)
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end

% save figure to file
if is_savetofile
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,10,12],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = sprintf('./figs/rank_regression_icar_parameters_error_%s',errortype);
    print(fname,'-painters','-dpdf');
end


%% Structural equation modeling

%   This part of the analysis was done in R. Check the file ./sem/sem_icar_sigma.R
%    for the script that runs the actual SEM.

%   The data in this analysis come from the subject indices
%       idx_incl_tt: 
%                If the inclusion criterion for hidden-state inference was based on 
%                   a p-value threshold for the null model (0.05, 0.01, and
%                   0.001) 
%                To reproduce the indices, set the value of 'nullp' to the
%                   desired p-value in the above cell titled,
%                   'Model-based Analyses of RLINF (REQ'D)'
%
%       idx_subj_concat(:,1) & ~isnan(pars(:,1)): 
%                If there is no criterion on hidden-state inference.
%
%   The data for all inclusion criteria are found in ./sem/dat/
%   The data were produced via the following code:

% ~~~~~~~~~~~~~~~~~~~~~ USER INPUT ~~~~~~~~~~~~~~~~~~~~~
is_nullp = true; % set to true if using the nullp threshold set above
% ~~~~~~~~~~~~~~~~~~~ END USER INPUT ~~~~~~~~~~~~~~~~~~~

if is_nullp
    fprintf('Using indices from hinf with p-value threshold %.3f\n',nullp);
    isok = idx_incl_tt;
    pstr = sprintf('p%03d',threshold*1000);
else
    isok = idx_subj_concat(:,1) & ~isnan(pars(:,1));
    pstr = 'pnon';
end

icar_sem_test   = icar(isok,1);
icar_sem_retest = icar(isok,2);

nok     = sum(isok);
nok_rt  = sum(~isnan(icar_sem_retest));
fprintf('N(test): %d\n',nok);
fprintf('N(retest): %d\n',nok_rt);

pars_sem_test = [pars(isok,:,1,1) pars(isok,:,2,1)];
pars_sem_retest = [pars(isok,:,1,2) pars(isok,:,2,2)];

% export matrices as CSV files for SEM
writematrix(icar_sem_test,  sprintf('./sem/dat/icar_sem_test_%s.csv',pstr));
writematrix(pars_sem_test,  sprintf('./sem/dat/pars_sem_test_%s.csv',pstr));
writematrix(icar_sem_retest,sprintf('./sem/dat/icar_sem_retest_%s.csv',pstr));
writematrix(pars_sem_retest,sprintf('./sem/dat/pars_sem_retest_%s.csv',pstr));





