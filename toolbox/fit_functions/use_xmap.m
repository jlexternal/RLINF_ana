function [out] = use_xmap(out)
%  USE_XMAP  Use posterior maximum as point-wise parameter estimates
%
%  Usage: [out] = USE_XMAP(out)
%
%  where out is a VBMC fitting output structure
%
%  Valentin Wyart <valentin.wyart@inserm.fr>

% check fitting output structure
if ~isfield(out,'xnam') || ~isfield(out,'xmap')
    error('Invalid VBMC fitting output structure!');
end

% use posterior maximum as point-wise parameter estimates
for ifit = 1:out.nfit
    out.(out.xnam{ifit}) = out.xmap(ifit);
end

end