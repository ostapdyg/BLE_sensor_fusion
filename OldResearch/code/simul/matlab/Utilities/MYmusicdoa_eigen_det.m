function [eigenvects,eigenvals,elSpacing, scanAng] = MYmusicdoa_eigen_det(R, Nsig, varargin)
%musicdoa   MUSIC direction of arrival (DOA)

% phased.internal.narginchk(2,6,nargin);
phased.internal.narginchk(3,7,nargin);

[elSpacing, scanAng, eigenvects, diagEigenVals]= ...
    parseInput(R,Nsig,varargin{:});

% Sort eigenvectors
[~,indx1] = sort(diagEigenVals,'descend');
 eigenvects = eigenvects(:,indx1);
 eigenvals = diagEigenVals(indx1);


%--------------------------------------
function [elSpacing,scanAng,eigenvects,diagEigenVals] = ...
    parseInput(R,numSignals,varargin)

eml_assert_no_varsize(1:nargin, R, numSignals, varargin{:});
validateattributes(R,{'double'},{'finite','square'},...
        'musicdoa','R');
tol = 10*eps(max(abs(diag(R))));   % based on stats cholcov
cond = any(any(abs(R - R') > tol));
if cond
    coder.internal.errorIf(cond,...
         'phased:phased:notHermitian','R');
end

% Check for positive semi definite
[eigenvects,sEDArg] = eig((R+R')/2);  % ensure Hermitian
sED = diag(real(sEDArg));
diagEigenVals =sED;
tol = eps(max(abs(sED))); % based on stats cholcov
sED(abs(sED)<=tol)=0;
cond = any(sED<0);
if cond
    coder.internal.errorIf(cond,...
         'phased:phased:notPositiveSemiDefinite','R');
end
                   
M = size(R,1);

validateattributes(numSignals,{'double'},...
    {'scalar','positive','finite','integer','<',M},'musicdoa','NSIG');

defaultElSpacing = 0.5;
defaultScanAng = -90:90;

if isempty(coder.target)
     p = inputParser;
     p.addParameter('ElementSpacing',defaultElSpacing);
     p.addParameter('ScanAngles',defaultScanAng);
     p.parse(varargin{:});
     elSpacing = p.Results.ElementSpacing;
     scanAng = p.Results.ScanAngles;
else
    parms = struct( ...
        'ElementSpacing',uint32(0), ...
        'ScanAngles',uint32(0));
    poptions = struct( ...
        'PartialMatching','unique', ...
        'StructExpand',false);
    pstruct = coder.internal.parseParameterInputs(parms,poptions,varargin{:});
    elSpacing = coder.internal.getParameterValue(...
      pstruct.ElementSpacing,defaultElSpacing,varargin{:});
    scanAng = coder.internal.getParameterValue(...
      pstruct.ScanAngles,defaultScanAng,varargin{:});
end

validateattributes(elSpacing,{'double'},...
  {'scalar','positive','finite'},'musicdoa','DIST');

validateattributes(scanAng,{'double'},...
  {'vector','finite','<=',90,'>=',-90},'musicdoa','SCANANG');

scanAng = scanAng(:)';

% [EOF]