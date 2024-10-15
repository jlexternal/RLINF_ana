function merge_model_fits(pathname,fun_fit,samp,sfit)
%  MERGE_MODEL_FITS  Merge individual model fits to single MAT file
%
%  Usage: MERGE_MODEL_FITS(pathname,fun_fit,samp,sfit)
%
%  where pathname is the path to individual model fits
%        fun_fit is the name of the fitted model
%        samp is the sample name
%        sfit is the model-specific fitting configuration string
%
%  All input arguments are required. The function will ask whether to merge
%  individual fits to single MAT file, and whether to delete 
%
%  Valentin Wyart <valentin.wyart@inserm.fr>

% check number of input arguments
narginchk(4,4);

% set list of conditions (0=bandit, 1=fairy)
condlist = [0,1];

% get total number of subjects in sample
fname = sprintf('./processed/%s/preprocessed_data_%s.mat',samp,samp);
if ~exist(fname,'file')
    error('Datafile not found!');
end
load(fname,'idx_fb');
nsubj = size(idx_fb,1);

% merge BADS fits
out_bads = cell(nsubj,2);
for icond = 1:2
    cond = condlist(icond);
    % identify fitting output files
    d = dir(fullfile(pathname,sprintf('%s_%s_subj*_cond%d_%s_bads.mat',fun_fit,samp,cond,sfit)));
    nfile = numel(d);
    % load and merge fits into cell array
    for ifile = 1:nfile
        A = sscanf(d(ifile).name,sprintf('%s_%s_subj%%03d_cond%%d_%s_bads.mat',fun_fit,samp,sfit));
        isubj = A(1);
        load(fullfile(d(ifile).folder,d(ifile).name),'out');
        out_bads{isubj,icond} = out;
    end
end

% merge VBMC fits
out_vbmc = cell(nsubj,2);
for icond = 1:2
    cond = condlist(icond);
    % identify fitting output files
    d = dir(fullfile(pathname,sprintf('%s_%s_subj*_cond%d_%s_vbmc.mat',fun_fit,samp,cond,sfit)));
    nfile = numel(d);
    % load and merge fits into cell array
    for ifile = 1:nfile
        A = sscanf(d(ifile).name,sprintf('%s_%s_subj%%03d_cond%%d_%s_vbmc.mat',fun_fit,samp,sfit));
        isubj = A(1);
        load(fullfile(d(ifile).folder,d(ifile).name),'out');
        out_vbmc{isubj,icond} = out;
    end
end

% check for match between BADS and VBMC fits
isok_bads = ~cellfun(@isempty,out_bads);
isok_vbmc = ~cellfun(@isempty,out_vbmc);
if ~all(isok_bads == isok_vbmc,'all')
    error('Mismatch between BADS and VBMC fits!');
end

fprintf('%d subjects with fit in bandit task.\n',sum(isok_vbmc(:,1)));
fprintf('%d subjects with fit in fairy task.\n',sum(isok_vbmc(:,2)));
fprintf('%d subjects with fits in both tasks.\n',sum(all(isok_vbmc,2)));

% save individual fits to file
merged = false;
switch input('Merge individual fits to file? (yes/no)\n','s')
    case 'yes'
        dname = './fit/merged';
        fname = sprintf('%s_%s_%s.mat',fun_fit,samp,sfit);
        save(fullfile(dname,fname),'condlist','out_bads','out_vbmc');
        fprintf('Saved fits to %s\n\n',fullfile(dname,fname));
        merged = true;
    otherwise
        fprintf('Did not save fits to file.\n');
end
if ~merged
    return
end

% delete individual fits
switch input('Delete individual fits? (yes/no)\n','s')
    case 'yes'
        delete(fullfile(pathname,sprintf('%s_%s_subj*_cond%d_%s_*.mat',fun_fit,samp,cond,sfit)));
        fprintf('Deleted individual fits.\n');
    otherwise
        fprintf('Did not delete individual fits.\n');
end

end