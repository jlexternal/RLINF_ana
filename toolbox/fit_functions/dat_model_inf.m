function [dat] = dat_model_inf(samp,subj,cond)
%  DAT_MODEL_INF  Get data structure for fitting inference model
%
%  Usage: [dat] = DAT_MODEL_INF(samp,subj,cond)
%
%  where samp is the sample name
%        subj is the subject number
%        cond is the condition number (0=bandit, 1=fairy)
%
%  Valentin Wyart <valentin.wyart@inserm.fr>

% check input arguments
narginchk(3,3);
if ~ischar(samp)
    error('Invalid sample name!');
end
if ~isnumeric(subj) || ~isscalar(subj) || rem(subj,1) ~= 0
    error('Invalid subject number!');
end
if ~isnumeric(cond) || ~isscalar(cond) || ~ismember(cond,[0,1])
    error('Invalid condition number (0=bandit, 1=fairy)!');
end

% load datafile
fname = sprintf('./processed/%s/preprocessed_data_%s.mat',samp,samp);
if ~exist(fname,'file')
    error('Datafile not found!');
end
load(fname);

% compute evidence scaling factor
perr = 0.3; % overlap between generative distributions
x = -1:0.001:+1;
efac = fzero(@(b)trapz(x(x > 0),1./(1+exp(x(x > 0)*b))/trapz(x,1./(1+exp(x*b))))-perr,[0,10]);

% get task variables and behavior
ifilt = idx_cond(subj,:) == cond;

rts     = idx_rt(subj,ifilt);
blknum  = idx_blk(subj,ifilt);
trlnum  = repmat(1:nnz(blknum == 1),[1,numel(unique(blknum))]);
bmstate = 2-idx_bmstate(subj,ifilt);
fbcorr  = idx_fbabs(subj,ifilt)/100*2-1;
fbboth  = cat(1,+fbcorr,-fbcorr);
stim    = fbboth(sub2ind(size(fbboth),bmstate,1:numel(bmstate)));
resp    = 2-idx_blmn(subj,ifilt);
iscor   = idx_corr(subj,ifilt);
clear('fbcorr','fbboth');

% realign the two conditions wrt common frame
% response (resp) now follows stimulus (stim) in both conditions
if cond == 0
    % bandit task
    itrl = find(trlnum <= 72);
    blknum = blknum(itrl);
    trlnum = trlnum(itrl);
    bmstate = bmstate(itrl);
    stim = stim(itrl);
    rprv = resp(itrl);
    resp = resp(itrl+1);
    rts  = rts(itrl);
else
    % fairy task
    itrl = find(trlnum >= 2);
    blknum = blknum(itrl);
    trlnum = trlnum(itrl)-1;
    bmstate = bmstate(itrl);
    stim = stim(itrl);
    rprv = resp(itrl-1);
    resp = resp(itrl);
    rts  = rts(itrl);
end

% create data structure
dat         = [];
dat.samp    = samp;       % sample name
dat.subj    = subj;       % subject number
dat.cond    = cond;       % condition number (0=bandit, 1=fairy)
dat.trlnum  = trlnum(:);  % trial number in current block
dat.bmstate = bmstate(:); % hidden state direction
dat.stim    = stim(:);    % stimulus direction
dat.resp    = resp(:);    % current response direction
dat.rprv    = rprv(:);    % previous response direction
dat.efac    = efac;       % evidence scaling factor
dat.rts     = rts;        % reaction times

end