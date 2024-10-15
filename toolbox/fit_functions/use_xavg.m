function [out] = use_xavg(out)
%  USE_XAVG  Use posterior average as point-wise parameter estimates
%
%  Usage: [out] = USE_XAVG(out)
%
%  where out is a VBMC fitting output structure
%
%  Valentin Wyart <valentin.wyart@inserm.fr>

% check fitting output structure
if ~isfield(out,'xnam') || ~isfield(out,'xavg')
    error('Invalid VBMC fitting output structure!');
end

% use posterior average as point-wise parameter estimates
for ifit = 1:out.nfit
    out.(out.xnam{ifit}) = out.xavg(ifit);
end

end